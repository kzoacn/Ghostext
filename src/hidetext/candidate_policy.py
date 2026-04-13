from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .config import CandidatePolicyConfig
from .model_backend import RawNextTokenDistribution, TextBackend, TokenProb


@dataclass(frozen=True)
class CandidateSelection:
    entries: tuple[TokenProb, ...]
    entropy_bits: float
    allows_encoding: bool

    @property
    def top(self) -> TokenProb:
        return self.entries[0]


def select_candidates(
    distribution: list[TokenProb] | RawNextTokenDistribution,
    config: CandidatePolicyConfig,
    *,
    backend: TextBackend | None = None,
) -> CandidateSelection:
    if isinstance(distribution, RawNextTokenDistribution):
        if backend is None:
            raise ValueError("backend is required for raw distributions")
        return _select_from_raw_distribution(distribution, config, backend=backend)
    return _select_from_probs(distribution, config)


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
