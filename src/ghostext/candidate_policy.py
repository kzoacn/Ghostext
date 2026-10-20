from __future__ import annotations

from dataclasses import dataclass, replace
import math
import numpy as np

from .config import CandidatePolicyConfig
from .errors import ModelBackendError, UnsafeTokenizationError
from .model_backend import RawNextTokenDistribution, TextBackend, TokenProb


@dataclass(frozen=True)
class CandidateSelection:
    entries: tuple[TokenProb, ...]
    entropy_bits: float
    allows_encoding: bool

    @property
    def top(self) -> TokenProb:
        return self.entries[0]


@dataclass(frozen=True)
class CandidateAudit:
    prefilter_selection: CandidateSelection
    stable_selection: CandidateSelection
    truncated_tail_mass: float
    stability_removed_mass: float
    fallback_used: bool = False


def audit_candidate_selection(
    distribution: list[TokenProb] | RawNextTokenDistribution,
    config: CandidatePolicyConfig,
    *,
    backend: TextBackend | None = None,
    prompt: str | None = None,
    generated_token_ids: list[int] | None = None,
) -> CandidateAudit:
    """Observe the actual policy without changing token-selection rules.

    Masses are measured before integer quantization, relative to the backend's
    already masked distribution and then its truncated distribution. When
    fallback_used, stability_removed_mass also includes the final stable-token
    cap: it is the total mass discarded after the expanded prefilter pool.
    """
    if isinstance(distribution, RawNextTokenDistribution):
        if backend is None:
            raise ValueError("backend is required for raw distributions")
        prefilter = _select_from_raw_distribution(distribution, config, backend=backend)
    else:
        prefilter = _select_from_probs(distribution, config)
    prefilter, stable, fallback = _finish_selection(
        distribution, prefilter, config, backend=backend, prompt=prompt,
        generated_token_ids=generated_token_ids,
    )
    selected_ids = {entry.token_id for entry in prefilter.entries}
    if isinstance(distribution, RawNextTokenDistribution):
        token_ids = np.asarray(distribution.token_ids)
        logits = np.asarray(distribution.logits, dtype=np.float64)
        mask = np.isin(token_ids, list(selected_ids))
        retained_mass = float(np.exp(logits[mask] - _logsumexp(logits)).sum())
    else:
        total = sum(entry.probability for entry in distribution)
        retained_mass = sum(
            entry.probability for entry in distribution if entry.token_id in selected_ids
        ) / total
    stable_ids = {entry.token_id for entry in stable.entries}
    removed_mass = sum(
        entry.probability for entry in prefilter.entries if entry.token_id not in stable_ids
    )
    return CandidateAudit(
        prefilter_selection=prefilter,
        stable_selection=stable,
        truncated_tail_mass=max(0.0, min(1.0, 1.0 - retained_mass)),
        stability_removed_mass=removed_mass,
        fallback_used=fallback,
    )


def select_candidates(
    distribution: list[TokenProb] | RawNextTokenDistribution,
    config: CandidatePolicyConfig,
    *,
    backend: TextBackend | None = None,
    prompt: str | None = None,
    generated_token_ids: list[int] | None = None,
) -> CandidateSelection:
    selection: CandidateSelection
    if isinstance(distribution, RawNextTokenDistribution):
        if backend is None:
            raise ValueError("backend is required for raw distributions")
        selection = _select_from_raw_distribution(distribution, config, backend=backend)
    else:
        selection = _select_from_probs(distribution, config)
    return _finish_selection(
        distribution, selection, config, backend=backend, prompt=prompt,
        generated_token_ids=generated_token_ids,
    )[1]


def _finish_selection(distribution, selection, config, *, backend, prompt, generated_token_ids):
    try:
        stable = _enforce_retokenization_stability(
            selection, config, backend=backend, prompt=prompt,
            generated_token_ids=generated_token_ids,
        )
        return selection, stable, False
    except UnsafeTokenizationError:
        if config.empty_candidate_search <= config.max_candidates:
            raise
    # This optional versioned policy changes the source only when the original
    # truncated set has no stable token. Scan a bounded larger rank-ordered
    # pool, ignoring top_p in this fallback, and retain at most max_candidates.
    expanded = replace(config, top_p=1.0, max_candidates=config.empty_candidate_search)
    if isinstance(distribution, RawNextTokenDistribution):
        pool = _select_from_raw_distribution(distribution, expanded, backend=backend)
    else:
        pool = _select_from_probs(distribution, expanded)
    stable = _enforce_retokenization_stability(
        pool, config, backend=backend, prompt=prompt, generated_token_ids=generated_token_ids,
        retain_limit=config.max_candidates,
    )
    return pool, stable, True


def _select_from_probs(
    distribution: list[TokenProb],
    config: CandidatePolicyConfig,
) -> CandidateSelection:
    if not distribution:
        raise ValueError("distribution must not be empty")

    ordered = sorted(
        distribution,
        key=lambda item: (-item.probability, item.token_id),
    )

    selected: list[TokenProb] = []
    cumulative = 0.0
    for token in ordered:
        selected.append(token)
        cumulative += token.probability
        if len(selected) >= config.max_candidates or cumulative >= config.top_p:
            break

    normalizer = sum(token.probability for token in selected)
    normalized = [
        TokenProb(
            token=token.token,
            token_id=token.token_id,
            probability=token.probability / normalizer,
        )
        for token in selected
    ]
    entropy_bits = -sum(
        token.probability * math.log2(token.probability)
        for token in normalized
        if token.probability > 0.0
    )
    allows_encoding = len(normalized) >= 2 and entropy_bits >= config.min_entropy_bits
    return CandidateSelection(
        entries=tuple(normalized),
        entropy_bits=entropy_bits,
        allows_encoding=allows_encoding,
    )


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    shifted = np.exp(values - max_value)
    return max_value + float(np.log(np.sum(shifted)))


def _select_from_raw_distribution(
    distribution: RawNextTokenDistribution,
    config: CandidatePolicyConfig,
    *,
    backend: TextBackend,
) -> CandidateSelection:
    token_ids = np.asarray(distribution.token_ids, dtype=np.int64)
    logits = np.asarray(distribution.logits, dtype=np.float64)
    if token_ids.size == 0 or logits.size == 0:
        raise ValueError("distribution must not be empty")
    if token_ids.shape != logits.shape:
        raise ValueError("token_ids and logits must have matching shapes")

    log_z = _logsumexp(logits)
    vocab_size = int(token_ids.size)
    max_candidates = min(config.max_candidates, vocab_size)
    probe_size = min(vocab_size, max(max_candidates * 8, 256))

    selected_indices: list[int] | None = None
    selected_raw_probs: list[float] | None = None
    while True:
        if probe_size == vocab_size:
            probe_indices = np.arange(vocab_size, dtype=np.int64)
        else:
            probe_indices = np.argpartition(logits, -probe_size)[-probe_size:]

        ordered = sorted(
            probe_indices.tolist(),
            key=lambda index: (-float(logits[index]), int(token_ids[index])),
        )
        raw_probs = np.exp(logits[ordered] - log_z)

        cumulative = 0.0
        current_indices: list[int] = []
        current_raw_probs: list[float] = []
        for index, raw_prob in zip(ordered, raw_probs, strict=True):
            current_indices.append(index)
            current_raw_probs.append(float(raw_prob))
            cumulative += float(raw_prob)
            if len(current_indices) >= max_candidates or cumulative >= config.top_p:
                break

        reached_limit = len(current_indices) >= max_candidates or cumulative >= config.top_p
        if reached_limit or probe_size == vocab_size:
            selected_indices = current_indices
            selected_raw_probs = current_raw_probs
            break
        probe_size = min(vocab_size, probe_size * 2)

    assert selected_indices is not None
    assert selected_raw_probs is not None

    selected_mass = sum(selected_raw_probs)
    normalized = [
        TokenProb(
            token=backend.token_text(int(token_ids[index])),
            token_id=int(token_ids[index]),
            probability=raw_prob / selected_mass,
        )
        for index, raw_prob in zip(selected_indices, selected_raw_probs, strict=True)
    ]
    entropy_bits = -sum(
        token.probability * math.log2(token.probability)
        for token in normalized
        if token.probability > 0.0
    )
    allows_encoding = len(normalized) >= 2 and entropy_bits >= config.min_entropy_bits
    return CandidateSelection(
        entries=tuple(normalized),
        entropy_bits=entropy_bits,
        allows_encoding=allows_encoding,
    )


def _enforce_retokenization_stability(
    selection: CandidateSelection,
    config: CandidatePolicyConfig,
    *,
    backend: TextBackend | None,
    prompt: str | None,
    generated_token_ids: list[int] | None,
    retain_limit: int | None = None,
) -> CandidateSelection:
    if not config.enforce_retokenization_stability:
        return selection
    if backend is None or prompt is None or generated_token_ids is None:
        return selection

    stable_entries = []
    for entry in selection.entries:
        if _is_retokenization_stable(
            backend,
            prompt=prompt,
            generated_token_ids=generated_token_ids,
            token_id=entry.token_id,
        ):
            stable_entries.append(entry)
            if retain_limit is not None and len(stable_entries) >= retain_limit:
                break
    if not stable_entries:
        raise UnsafeTokenizationError(
            "candidate selection contains no retokenization-stable token"
        )

    total_probability = sum(entry.probability for entry in stable_entries)
    normalized = [
        TokenProb(
            token=entry.token,
            token_id=entry.token_id,
            probability=entry.probability / total_probability,
        )
        for entry in stable_entries
    ]
    entropy_bits = -sum(
        entry.probability * math.log2(entry.probability)
        for entry in normalized
        if entry.probability > 0.0
    )
    allows_encoding = len(normalized) >= 2 and entropy_bits >= config.min_entropy_bits
    return CandidateSelection(
        entries=tuple(normalized),
        entropy_bits=entropy_bits,
        allows_encoding=allows_encoding,
    )


def _is_retokenization_stable(
    backend: TextBackend,
    *,
    prompt: str,
    generated_token_ids: list[int],
    token_id: int,
) -> bool:
    candidate_token_ids = [*generated_token_ids, token_id]
    try:
        rendered = backend.render(candidate_token_ids)
        retokenized = backend.tokenize(rendered, prompt)
    except ModelBackendError:
        return False
    return retokenized == candidate_token_ids
