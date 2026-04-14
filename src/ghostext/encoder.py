from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
import random
import sys
from time import perf_counter
from typing import Callable

from .codec import MessageSegmentEncoder
from .config import RuntimeConfig
from .crypto import build_packet
from .errors import (
    EncodingExhaustedError,
    LowEntropyRetryLimitError,
    StallDetectedError,
    UnsafeTokenizationError,
)
from .model_backend import TextBackend
from .packet import packet_bootstrap_size
from .pipeline import prepare_quantized_distribution
from .progress import ProgressCallback, ProgressSnapshot

STALL_PROGRESS_EPSILON_BITS = 1e-9
RetryNoticeCallback = Callable[[str], None]


@dataclass(frozen=True)
class SegmentStats:
    name: str
    tokens_used: int
    encoding_steps: int
    embedded_bits: float


@dataclass(frozen=True)
class EncodeResult:
    text: str
    token_ids: tuple[int, ...]
    packet: bytes
    config_fingerprint: int
    segment_stats: tuple[SegmentStats, ...]
    attempts_used: int
    elapsed_seconds: float

    @property
    def total_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def packet_tokens(self) -> int:
        return sum(segment.tokens_used for segment in self.segment_stats)

    @property
    def tail_tokens(self) -> int:
        return self.total_tokens - self.packet_tokens

    @property
    def bits_per_token(self) -> float:
        if not self.token_ids:
            return 0.0
        return (len(self.packet) * 8) / len(self.token_ids)

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0.0:
            return 0.0
        return self.total_tokens / self.elapsed_seconds


@dataclass
class _LowEntropyWindow:
    window_tokens: int
    threshold_bits: float
    values: deque[float] = field(default_factory=deque)
    total_entropy_bits: float = 0.0

    def observe(self, entropy_bits: float) -> float | None:
        if self.window_tokens <= 0:
            return None
        self.values.append(entropy_bits)
        self.total_entropy_bits += entropy_bits
        if len(self.values) > self.window_tokens:
            self.total_entropy_bits -= self.values.popleft()
        if len(self.values) < self.window_tokens:
            return None
        average_entropy_bits = self.total_entropy_bits / self.window_tokens
        if average_entropy_bits < self.threshold_bits:
            return average_entropy_bits
        return None


class _LowEntropyWindowTriggered(Exception):
    def __init__(self, *, segment_name: str, average_entropy_bits: float) -> None:
        super().__init__(segment_name, average_entropy_bits)
        self.segment_name = segment_name
        self.average_entropy_bits = average_entropy_bits


class StegoEncoder:
    def __init__(self, backend: TextBackend, config: RuntimeConfig | None = None) -> None:
        self.backend = backend
        self.config = config or RuntimeConfig()

    def encode(
        self,
        plaintext: str,
        *,
        passphrase: str,
        prompt: str,
        salt: bytes | None = None,
        nonce: bytes | None = None,
        progress_callback: ProgressCallback | None = None,
        retry_notice_callback: RetryNoticeCallback | None = None,
    ) -> EncodeResult:
        start_time = perf_counter()
        plaintext_bytes = plaintext.encode("utf-8")
        config_fingerprint = self.config.config_fingerprint(
            backend_metadata=self.backend.metadata.as_dict(),
            prompt=prompt,
        )
        attempts_allowed = max(1, self.config.codec.max_encode_attempts)
        can_retry = salt is None or nonce is None
        if not can_retry:
            attempts_allowed = 1

        for attempt_index in range(attempts_allowed):
            packet = build_packet(
                plaintext_bytes,
                passphrase=passphrase,
                config_fingerprint=config_fingerprint,
                crypto_config=self.config.crypto,
                salt=salt,
                nonce=nonce,
            )
            try:
                token_ids, stats = self._encode_packet(
                    packet=packet,
                    prompt=prompt,
                    start_time=start_time,
                    progress_callback=progress_callback,
                )
            except _LowEntropyWindowTriggered as exc:
                if attempt_index + 1 < attempts_allowed:
                    self._emit_retry_notice(
                        attempt_index=attempt_index + 1,
                        attempts_allowed=attempts_allowed,
                        reason=(
                            "low entropy "
                            f"({exc.segment_name}, avg={exc.average_entropy_bits:.3f} bits)"
                        ),
                        retry_notice_callback=retry_notice_callback,
                    )
                    continue
                raise LowEntropyRetryLimitError(
                    "encoding entered a low-entropy regime: "
                    f"{exc.segment_name} rolling average entropy stayed at "
                    f"{exc.average_entropy_bits:.3f} bits over "
                    f"{self.config.codec.low_entropy_window_tokens} consecutive tokens "
                    f"across {attempt_index + 1} attempt(s). {self._retry_hint(can_retry)}"
                ) from exc
            except UnsafeTokenizationError as exc:
                if attempt_index + 1 < attempts_allowed:
                    self._emit_retry_notice(
                        attempt_index=attempt_index + 1,
                        attempts_allowed=attempts_allowed,
                        reason="no retokenization-stable candidate path",
                        retry_notice_callback=retry_notice_callback,
                    )
                    continue
                raise UnsafeTokenizationError(
                    "encoding encountered a retokenization-unstable candidate set and "
                    f"could not find a safe token path across {attempt_index + 1} "
                    f"attempt(s). {self._retry_hint(can_retry)}"
                ) from exc
            self._extend_with_natural_tail(
                packet=packet,
                prompt=prompt,
                generated_token_ids=token_ids,
                attempt_index=attempt_index,
            )

            return EncodeResult(
                text=self.backend.render(token_ids),
                token_ids=tuple(token_ids),
                packet=packet,
                config_fingerprint=config_fingerprint,
                segment_stats=tuple(stats),
                attempts_used=attempt_index + 1,
                elapsed_seconds=perf_counter() - start_time,
            )

        raise AssertionError("encoding attempts loop exited unexpectedly")

    def _encode_packet(
        self,
        *,
        packet: bytes,
        prompt: str,
        start_time: float,
        progress_callback: ProgressCallback | None,
    ) -> tuple[list[int], list[SegmentStats]]:
        token_ids: list[int] = []
        stats: list[SegmentStats] = []
        total_bits = len(packet) * 8
        completed_bits = 0.0
        low_entropy_window = _LowEntropyWindow(
            window_tokens=self.config.codec.low_entropy_window_tokens,
            threshold_bits=self.config.codec.low_entropy_threshold_bits,
        )

        bootstrap_size = packet_bootstrap_size(
            self.config.crypto.salt_len,
            self.config.crypto.nonce_len,
        )
        header_stats = self._encode_segment(
            segment_name="header",
            payload=packet[:bootstrap_size],
            prompt=prompt,
            generated_token_ids=token_ids,
            max_tokens=self.config.codec.max_header_tokens,
            completed_bits_before=completed_bits,
            overall_bits_total=total_bits,
            start_time=start_time,
            progress_callback=progress_callback,
            low_entropy_window=low_entropy_window,
        )
        stats.append(header_stats)
        completed_bits += bootstrap_size * 8

        body_stats = self._encode_segment(
            segment_name="body",
            payload=packet[bootstrap_size:],
            prompt=prompt,
            generated_token_ids=token_ids,
            max_tokens=self.config.codec.max_body_tokens,
            completed_bits_before=completed_bits,
            overall_bits_total=total_bits,
            start_time=start_time,
            progress_callback=progress_callback,
            low_entropy_window=low_entropy_window,
        )
        stats.append(body_stats)
        return token_ids, stats

    def _extend_with_natural_tail(
        self,
        *,
        packet: bytes,
        prompt: str,
        generated_token_ids: list[int],
        attempt_index: int,
    ) -> None:
        max_tail_tokens = self.config.codec.natural_tail_max_tokens
        if max_tail_tokens <= 0:
            return

        recent_text = self.backend.render(generated_token_ids)[-64:]
        if self._looks_naturally_finished(recent_text):
            return

        rng = random.Random(self._natural_tail_seed(packet=packet, attempt_index=attempt_index))
        for _ in range(max_tail_tokens):
            try:
                quantized = prepare_quantized_distribution(
                    self.backend,
                    prompt=prompt,
                    generated_token_ids=generated_token_ids,
                    config=self.config,
                )
            except UnsafeTokenizationError:
                break
            generated_token_ids.append(self._sample_tail_token_id(quantized, rng))
            recent_text = self.backend.render(generated_token_ids)[-64:]
            if self._looks_naturally_finished(recent_text):
                break

    def _natural_tail_seed(self, *, packet: bytes, attempt_index: int) -> int:
        payload = (
            self.config.seed.to_bytes(8, "big", signed=True)
            + attempt_index.to_bytes(4, "big", signed=False)
            + packet
        )
        return int.from_bytes(sha256(payload).digest()[:8], "big")

    def _sample_tail_token_id(self, distribution, rng: random.Random) -> int:
        if len(distribution.entries) == 1:
            return distribution.top.token_id
        draw = rng.random()
        cumulative = 0.0
        for entry in distribution.entries:
            cumulative += entry.probability
            if draw <= cumulative:
                return entry.token_id
        return distribution.entries[-1].token_id

    def _retry_hint(self, can_retry: bool) -> str:
        retry_hint = "Try a different prompt or reduce the message length."
        if not can_retry:
            return "Automatic retry is disabled when both salt and nonce are fixed. " + retry_hint
        return retry_hint

    def _emit_retry_notice(
        self,
        *,
        attempt_index: int,
        attempts_allowed: int,
        reason: str,
        retry_notice_callback: RetryNoticeCallback | None,
    ) -> None:
        if retry_notice_callback is not None:
            retry_notice_callback(
                f"[retry {attempt_index}/{attempts_allowed}] restarting with fresh packet: {reason}"
            )
            return
        print(
            f"[ghostext][retry {attempt_index}/{attempts_allowed}] {reason}",
            file=sys.stderr,
            flush=True,
        )

    def _looks_naturally_finished(self, text: str) -> bool:
        trimmed = text.rstrip()
        if not trimmed:
            return False
        return (
            trimmed.endswith((".", "!", "?", "。", "！", "？"))
            or trimmed.endswith("\n\n")
        )

    def _encode_segment(
        self,
        *,
        segment_name: str,
        payload: bytes,
        prompt: str,
        generated_token_ids: list[int],
        max_tokens: int,
        completed_bits_before: float,
        overall_bits_total: int,
        start_time: float,
        progress_callback: ProgressCallback | None,
        low_entropy_window: _LowEntropyWindow,
    ) -> SegmentStats:
        encoder = MessageSegmentEncoder(payload)
        steps = 0
        encoding_steps = 0
        embedded_bits = 0.0
        previous_resolved_bits = encoder.resolved_bits
        stall_run = 0
        while not encoder.finished:
            if steps >= max_tokens:
                raise EncodingExhaustedError(
                    f"{segment_name} segment exceeded token budget {max_tokens}"
                )
            quantized = prepare_quantized_distribution(
                self.backend,
                prompt=prompt,
                generated_token_ids=generated_token_ids,
                config=self.config,
            )
            low_entropy_average = low_entropy_window.observe(quantized.entropy_bits)
            if low_entropy_average is not None:
                raise _LowEntropyWindowTriggered(
                    segment_name=segment_name,
                    average_entropy_bits=low_entropy_average,
                )
            if quantized.allows_encoding:
                index, gained_bits = encoder.choose(quantized)
                chosen_token_id = quantized.entries[index].token_id
                encoding_steps += 1
                embedded_bits += gained_bits
            else:
                chosen_token_id = quantized.top.token_id
            generated_token_ids.append(chosen_token_id)
            steps += 1
            current_resolved_bits = encoder.resolved_bits
            if current_resolved_bits <= previous_resolved_bits + STALL_PROGRESS_EPSILON_BITS:
                stall_run += 1
            else:
                stall_run = 0
            if (
                self.config.codec.stall_patience_tokens > 0
                and stall_run >= self.config.codec.stall_patience_tokens
            ):
                raise StallDetectedError(
                    f"{segment_name} segment stalled for {stall_run} consecutive tokens "
                    f"at {current_resolved_bits:.3f}/{encoder.total_bits} bits"
                )
            previous_resolved_bits = current_resolved_bits
            self._emit_progress(
                segment_name=segment_name,
                encoder=encoder,
                segment_tokens=steps,
                generated_token_ids=generated_token_ids,
                max_tokens=max_tokens,
                completed_bits_before=completed_bits_before,
                overall_bits_total=overall_bits_total,
                start_time=start_time,
                progress_callback=progress_callback,
            )

        return SegmentStats(
            name=segment_name,
            tokens_used=steps,
            encoding_steps=encoding_steps,
            embedded_bits=embedded_bits,
        )

    def _emit_progress(
        self,
        *,
        segment_name: str,
        encoder: MessageSegmentEncoder,
        segment_tokens: int,
        generated_token_ids: list[int],
        max_tokens: int,
        completed_bits_before: float,
        overall_bits_total: int,
        start_time: float,
        progress_callback: ProgressCallback | None,
    ) -> None:
        if progress_callback is None:
            return
        elapsed_seconds = perf_counter() - start_time
        total_tokens = len(generated_token_ids)
        overall_bits_done = completed_bits_before + encoder.resolved_bits
        tokens_per_second = total_tokens / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        bits_per_token = overall_bits_done / total_tokens if total_tokens > 0 else 0.0
        progress_callback(
            ProgressSnapshot(
                phase="encode",
                segment_name=segment_name,
                segment_tokens=segment_tokens,
                total_tokens=total_tokens,
                token_budget=max_tokens,
                segment_bits_done=encoder.resolved_bits,
                segment_bits_total=encoder.total_bits,
                overall_bits_done=overall_bits_done,
                overall_bits_total=overall_bits_total,
                elapsed_seconds=elapsed_seconds,
                tokens_per_second=tokens_per_second,
                bits_per_token=bits_per_token,
                finished=encoder.finished,
            )
        )
