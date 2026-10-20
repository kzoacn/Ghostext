from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import secrets

from .quantization import QuantizedDistribution


def _interval_subrange(
    low: int,
    high: int,
    cdf_low: int,
    cdf_high: int,
    total_frequency: int,
) -> tuple[int, int]:
    width = high - low
    sub_low = low + (width * cdf_low) // total_frequency
    sub_high = low + (width * cdf_high) // total_frequency
    return sub_low, sub_high


@dataclass
class MessageSegmentEncoder:
    payload: bytes
    guard_bits: int = 0
    guard_value: int | None = None

    def __post_init__(self) -> None:
        _check_guard_bits(self.guard_bits)
        self.length = len(self.payload)
        self.total_bits = 8 * self.length
        self.domain = 1 << (self.total_bits + self.guard_bits)
        guard = self.guard_value
        if guard is None:
            guard = secrets.randbits(self.guard_bits)
        if not isinstance(guard, int) or not 0 <= guard < (1 << self.guard_bits):
            raise ValueError("guard_value does not fit guard_bits")
        self.target = (int.from_bytes(self.payload, "big") << self.guard_bits) | guard
        self.low = 0
        self.high = self.domain

    @property
    def finished(self) -> bool:
        return _prefix_resolved(self.low, self.high, self.guard_bits)

    @property
    def resolved_bits(self) -> float:
        if self.total_bits == 0:
            return 0.0
        if self.high <= self.low:
            raise ValueError("encoder interval collapsed")
        possible_payloads = ((self.high - 1) >> self.guard_bits) - (self.low >> self.guard_bits) + 1
        return max(0.0, self.total_bits - math.log2(possible_payloads))

    def choose(self, distribution: QuantizedDistribution) -> tuple[int, float]:
        if self.finished:
            raise ValueError("cannot encode an already resolved segment")
        resolved_before = self.resolved_bits
        for index, entry in enumerate(distribution.entries):
            sub_low, sub_high = _interval_subrange(
                self.low,
                self.high,
                entry.cdf_low,
                entry.cdf_high,
                distribution.total_frequency,
            )
            if sub_low <= self.target < sub_high:
                self.low = sub_low
                self.high = sub_high
                gained_bits = self.resolved_bits - resolved_before
                return index, gained_bits
        raise ValueError("target did not fall into any subinterval")


@dataclass
class MessageSegmentDecoder:
    payload_len: int
    guard_bits: int = 0

    def __post_init__(self) -> None:
        _check_guard_bits(self.guard_bits)
        if self.payload_len < 0:
            raise ValueError("payload_len must be nonnegative")
        self.total_bits = 8 * self.payload_len
        self.domain = 1 << (self.total_bits + self.guard_bits)
        self.low = 0
        self.high = self.domain

    @property
    def finished(self) -> bool:
        return _prefix_resolved(self.low, self.high, self.guard_bits)

    @property
    def resolved_bits(self) -> float:
        if self.total_bits == 0:
            return 0.0
        if self.high <= self.low:
            raise ValueError("decoder interval collapsed")
        possible_payloads = ((self.high - 1) >> self.guard_bits) - (self.low >> self.guard_bits) + 1
        return max(0.0, self.total_bits - math.log2(possible_payloads))

    def absorb(self, distribution: QuantizedDistribution, index: int) -> None:
        if self.finished:
            raise ValueError("cannot decode an already resolved segment")
        entry = distribution.entries[index]
        low, high = _interval_subrange(
            self.low,
            self.high,
            entry.cdf_low,
            entry.cdf_high,
            distribution.total_frequency,
        )
        if low >= high:
            raise ValueError("decoder interval collapsed")
        self.low, self.high = low, high

    def to_bytes(self) -> bytes:
        if not self.finished:
            raise ValueError("decoder interval is not yet resolved")
        if self.payload_len == 0:
            return b""
        return (self.low >> self.guard_bits).to_bytes(self.payload_len, "big")


def _check_guard_bits(guard_bits: int) -> None:
    if not isinstance(guard_bits, int) or guard_bits < 0:
        raise ValueError("guard_bits must be a nonnegative integer")


def _prefix_resolved(low: int, high: int, guard_bits: int) -> bool:
    return low < high and (low >> guard_bits) == ((high - 1) >> guard_bits)


def interval_rounding_tv(
    low: int, high: int, distribution: QuantizedDistribution
) -> Fraction:
    """Exact TV between integer subinterval sampling and f/F.

    The source integer must be uniform on [low, high). This is a diagnostic
    for the codec alone; it does not assume ciphertexts have that distribution
    or account for retries, candidate filtering, or length/stopping policies.
    """
    width = high - low
    if width <= 0:
        raise ValueError("interval must be nonempty")
    total = distribution.total_frequency
    numerator = 0
    for entry in distribution.entries:
        a, b = _interval_subrange(low, high, entry.cdf_low, entry.cdf_high, total)
        numerator += abs(total * (b - a) - width * entry.frequency)
    return Fraction(numerator, 2 * width * total)
