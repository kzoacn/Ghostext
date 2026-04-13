from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .codec import MessageSegmentEncoder
from .config import RuntimeConfig
from .crypto import build_packet
from .errors import EncodingExhaustedError
from .model_backend import TextBackend
from .packet import HEADER_SIZE
from .pipeline import prepare_quantized_distribution
from .progress import ProgressCallback, ProgressSnapshot


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
    elapsed_seconds: float

    @property
    def total_tokens(self) -> int:
        return len(self.token_ids)

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
    ) -> EncodeResult:
        start_time = perf_counter()
        plaintext_bytes = plaintext.encode("utf-8")
        config_fingerprint = self.config.config_fingerprint(
            backend_metadata=self.backend.metadata.as_dict(),
            prompt=prompt,
        )
        packet = build_packet(
            plaintext_bytes,
            passphrase=passphrase,
            config_fingerprint=config_fingerprint,
            crypto_config=self.config.crypto,
            salt=salt,
            nonce=nonce,
        )

        token_ids: list[int] = []
        stats: list[SegmentStats] = []
        total_bits = len(packet) * 8
        completed_bits = 0.0

        header_stats = self._encode_segment(
            segment_name="header",
            payload=packet[:HEADER_SIZE],
            prompt=prompt,
            generated_token_ids=token_ids,
            max_tokens=self.config.codec.max_header_tokens,
            completed_bits_before=completed_bits,
            overall_bits_total=total_bits,
            start_time=start_time,
            progress_callback=progress_callback,
        )
        stats.append(header_stats)
        completed_bits += len(packet[:HEADER_SIZE]) * 8

        body_stats = self._encode_segment(
            segment_name="body",
            payload=packet[HEADER_SIZE:],
            prompt=prompt,
            generated_token_ids=token_ids,
            max_tokens=self.config.codec.max_body_tokens,
            completed_bits_before=completed_bits,
            overall_bits_total=total_bits,
            start_time=start_time,
            progress_callback=progress_callback,
        )
        stats.append(body_stats)

        return EncodeResult(
            text=self.backend.render(token_ids),
            token_ids=tuple(token_ids),
            packet=packet,
            config_fingerprint=config_fingerprint,
            segment_stats=tuple(stats),
            elapsed_seconds=perf_counter() - start_time,
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
    ) -> SegmentStats:
        encoder = MessageSegmentEncoder(payload)
        steps = 0
        encoding_steps = 0
        embedded_bits = 0.0
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
            if quantized.allows_encoding:
                index, gained_bits = encoder.choose(quantized)
                chosen_token_id = quantized.entries[index].token_id
                encoding_steps += 1
                embedded_bits += gained_bits
            else:
                chosen_token_id = quantized.top.token_id
            generated_token_ids.append(chosen_token_id)
            steps += 1
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
