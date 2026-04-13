from __future__ import annotations

from dataclasses import dataclass
import math

from .config import CandidatePolicyConfig
from .model_backend import TokenProb


@dataclass(frozen=True)
class CandidateSelection:
    entries: tuple[TokenProb, ...]
    entropy_bits: float
    allows_encoding: bool

    @property
    def top(self) -> TokenProb:
        return self.entries[0]


def select_candidates(
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

