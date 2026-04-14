from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ProgressSnapshot:
    phase: str
    segment_name: str
    segment_tokens: int
    total_tokens: int
    token_budget: int
    segment_bits_done: float
    segment_bits_total: int
    overall_bits_done: float
    overall_bits_total: int | None
    elapsed_seconds: float
    tokens_per_second: float
    bits_per_token: float
    finished: bool


ProgressCallback = Callable[[ProgressSnapshot], None]

