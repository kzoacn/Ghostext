from __future__ import annotations

from dataclasses import dataclass

from .candidate_policy import CandidateSelection
from .model_backend import TokenProb


@dataclass(frozen=True)
class QuantizedEntry:
    token: str
    token_id: int
    probability: float
    frequency: int
    cdf_low: int
    cdf_high: int


@dataclass(frozen=True)
class QuantizedDistribution:
    entries: tuple[QuantizedEntry, ...]
    total_frequency: int
    entropy_bits: float
    allows_encoding: bool

    @property
    def top(self) -> QuantizedEntry:
        return self.entries[0]

    def find_index(self, token: str) -> int:
        for index, entry in enumerate(self.entries):
            if entry.token == token:
                return index
        raise KeyError(token)


def _quantize_probabilities(tokens: tuple[TokenProb, ...], total_frequency: int) -> list[int]:
    if len(tokens) > total_frequency:
        raise ValueError("candidate count exceeds total_frequency")

    base = [1] * len(tokens)
    remaining = total_frequency - len(tokens)
    scaled = [token.probability * remaining for token in tokens]
    base = [current + int(value) for current, value in zip(base, scaled, strict=True)]
    leftover = total_frequency - sum(base)
    order = sorted(
        range(len(tokens)),
        key=lambda index: (
            -(scaled[index] - int(scaled[index])),
            tokens[index].token_id,
        ),
    )
    for index in order[:leftover]:
        base[index] += 1
    return base


def quantize_candidates(
    selection: CandidateSelection,
    total_frequency: int,
) -> QuantizedDistribution:
    freqs = _quantize_probabilities(selection.entries, total_frequency)
    cdf = 0
    entries: list[QuantizedEntry] = []
    for token, freq in zip(selection.entries, freqs, strict=True):
        entries.append(
            QuantizedEntry(
                token=token.token,
                token_id=token.token_id,
                probability=token.probability,
                frequency=freq,
                cdf_low=cdf,
                cdf_high=cdf + freq,
            )
        )
        cdf += freq

    if cdf != total_frequency:
        raise ValueError("quantized frequencies do not sum to total_frequency")

    return QuantizedDistribution(
        entries=tuple(entries),
        total_frequency=total_frequency,
        entropy_bits=selection.entropy_bits,
        allows_encoding=selection.allows_encoding,
    )

