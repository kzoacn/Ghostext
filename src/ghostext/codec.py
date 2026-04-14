from __future__ import annotations

from dataclasses import dataclass
import math

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

    def __post_init__(self) -> None:
        self.length = len(self.payload)
        self.total_bits = 8 * self.length
        self.domain = 1 if self.length == 0 else 1 << (8 * self.length)
        self.target = int.from_bytes(self.payload, "big") if self.length else 0
        self.low = 0
        self.high = self.domain

    @property
    def finished(self) -> bool:
        return self.high - self.low == 1

    @property
    def resolved_bits(self) -> float:
        if self.total_bits == 0:
            return 0.0
        width = self.high - self.low
        if width <= 0:
            raise ValueError("encoder interval collapsed")
        return max(0.0, min(float(self.total_bits), self.total_bits - math.log2(width)))

    def choose(self, distribution: QuantizedDistribution) -> tuple[int, float]:
        width_before = self.high - self.low
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
                width_after = self.high - self.low
                gained_bits = 0.0
                if width_after > 0 and width_after < width_before:
                    gained_bits = math.log2(width_before / width_after)
                return index, gained_bits
        raise ValueError("target did not fall into any subinterval")


@dataclass
class MessageSegmentDecoder:
    payload_len: int

    def __post_init__(self) -> None:
        self.total_bits = 8 * self.payload_len
        self.domain = 1 if self.payload_len == 0 else 1 << (8 * self.payload_len)
        self.low = 0
        self.high = self.domain

    @property
    def finished(self) -> bool:
        return self.high - self.low == 1

    @property
    def resolved_bits(self) -> float:
        if self.total_bits == 0:
            return 0.0
        width = self.high - self.low
        if width <= 0:
            raise ValueError("decoder interval collapsed")
        return max(0.0, min(float(self.total_bits), self.total_bits - math.log2(width)))

    def absorb(self, distribution: QuantizedDistribution, index: int) -> None:
        entry = distribution.entries[index]
        self.low, self.high = _interval_subrange(
            self.low,
            self.high,
            entry.cdf_low,
            entry.cdf_high,
            distribution.total_frequency,
        )
        if self.low >= self.high:
            raise ValueError("decoder interval collapsed")

    def to_bytes(self) -> bytes:
        if not self.finished:
            raise ValueError("decoder interval is not yet resolved")
        if self.payload_len == 0:
            return b""
        return self.low.to_bytes(self.payload_len, "big")
