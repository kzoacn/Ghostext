from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .codec import MessageSegmentDecoder
from .config import RuntimeConfig
from .crypto import decrypt_packet
from .errors import ConfigMismatchError, PacketError, SynchronizationError
from .model_backend import TextBackend
from .packet import HEADER_SIZE, PacketHeader
from .pipeline import prepare_quantized_distribution
from .progress import ProgressCallback, ProgressSnapshot


@dataclass(frozen=True)
class DecodeResult:
    plaintext_bytes: bytes
    plaintext: str
    packet: bytes
    token_ids: tuple[int, ...]
    header: PacketHeader
    consumed_tokens: int
    elapsed_seconds: float

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0.0:
            return 0.0
        return self.consumed_tokens / self.elapsed_seconds

    @property
    def bits_per_token(self) -> float:
        if self.consumed_tokens <= 0:
            return 0.0
        return (len(self.packet) * 8) / self.consumed_tokens


class StegoDecoder:
    def __init__(self, backend: TextBackend, config: RuntimeConfig | None = None) -> None:
        self.backend = backend
        self.config = config or RuntimeConfig()

    def decode(
        self,
        stego_text: str,
        *,
        passphrase: str,
        prompt: str,
        progress_callback: ProgressCallback | None = None,
    ) -> DecodeResult:
        start_time = perf_counter()
        observed_token_ids = self.backend.tokenize(stego_text, prompt)
        consumed_token_ids: list[int] = []
        cursor = 0

        header_bytes, cursor = self._decode_segment(
            payload_len=HEADER_SIZE,
            observed_token_ids=observed_token_ids,
            cursor=cursor,
            consumed_token_ids=consumed_token_ids,
            prompt=prompt,
            max_tokens=self.config.codec.max_header_tokens,
            segment_name="header",
            completed_bits_before=0.0,
            overall_bits_total=None,
            start_time=start_time,
            progress_callback=progress_callback,
        )
        header = PacketHeader.unpack(header_bytes)
        expected_fp = self.config.config_fingerprint(
            backend_metadata=self.backend.metadata.as_dict(),
            prompt=prompt,
        )
        if header.config_fingerprint != expected_fp:
            raise ConfigMismatchError("packet config fingerprint does not match runtime")

        body_len = header.body_len
        total_bits = (HEADER_SIZE + body_len) * 8
        body_bytes, cursor = self._decode_segment(
            payload_len=body_len,
            observed_token_ids=observed_token_ids,
            cursor=cursor,
            consumed_token_ids=consumed_token_ids,
            prompt=prompt,
            max_tokens=self.config.codec.max_body_tokens,
            segment_name="body",
            completed_bits_before=HEADER_SIZE * 8,
            overall_bits_total=total_bits,
            start_time=start_time,
            progress_callback=progress_callback,
        )
        if cursor != len(observed_token_ids):
            raise SynchronizationError("stego text contains trailing tokens after packet end")

        packet = header_bytes + body_bytes
        plaintext_bytes = decrypt_packet(
            packet,
            passphrase=passphrase,
            expected_config_fingerprint=expected_fp,
            crypto_config=self.config.crypto,
        )
        return DecodeResult(
            plaintext_bytes=plaintext_bytes,
            plaintext=plaintext_bytes.decode("utf-8"),
            packet=packet,
            token_ids=tuple(observed_token_ids),
            header=header,
            consumed_tokens=cursor,
            elapsed_seconds=perf_counter() - start_time,
        )

    def _decode_segment(
        self,
        *,
        payload_len: int,
        observed_token_ids: list[int],
        cursor: int,
        consumed_token_ids: list[int],
        prompt: str,
        max_tokens: int,
        segment_name: str,
        completed_bits_before: float,
        overall_bits_total: int | None,
        start_time: float,
        progress_callback: ProgressCallback | None,
    ) -> tuple[bytes, int]:
        decoder = MessageSegmentDecoder(payload_len)
        steps = 0
        while not decoder.finished:
            if steps >= max_tokens:
                raise SynchronizationError(
                    f"{segment_name} segment exceeded token budget {max_tokens}"
                )
            if cursor >= len(observed_token_ids):
                raise PacketError(f"stego text ended before {segment_name} segment resolved")

            quantized = prepare_quantized_distribution(
                self.backend,
                prompt=prompt,
                generated_token_ids=consumed_token_ids,
                config=self.config,
            )
            observed_token_id = observed_token_ids[cursor]
            observed_text = self.backend.token_text(observed_token_id)

            if quantized.allows_encoding:
                try:
                    index = quantized.find_token_id_index(observed_token_id)
                except KeyError as exc:
                    raise SynchronizationError(
                        f"observed token {observed_text!r} not in {segment_name} candidate set"
                    ) from exc
                try:
                    decoder.absorb(quantized, index)
                except ValueError as exc:
                    raise SynchronizationError(
                        f"{segment_name} interval collapsed while absorbing token {observed_text!r}"
                    ) from exc
            else:
                if observed_token_id != quantized.top.token_id:
                    raise SynchronizationError(
                        f"expected deterministic token {quantized.top.token!r}, got {observed_text!r}"
                    )

            consumed_token_ids.append(observed_token_id)
            cursor += 1
            steps += 1
            self._emit_progress(
                segment_name=segment_name,
                decoder=decoder,
                segment_tokens=steps,
                consumed_token_ids=consumed_token_ids,
                max_tokens=max_tokens,
                completed_bits_before=completed_bits_before,
                overall_bits_total=overall_bits_total,
                start_time=start_time,
                progress_callback=progress_callback,
            )

        return decoder.to_bytes(), cursor

    def _emit_progress(
        self,
        *,
        segment_name: str,
        decoder: MessageSegmentDecoder,
        segment_tokens: int,
        consumed_token_ids: list[int],
        max_tokens: int,
        completed_bits_before: float,
        overall_bits_total: int | None,
        start_time: float,
        progress_callback: ProgressCallback | None,
    ) -> None:
        if progress_callback is None:
            return
        elapsed_seconds = perf_counter() - start_time
        total_tokens = len(consumed_token_ids)
        overall_bits_done = completed_bits_before + decoder.resolved_bits
        tokens_per_second = total_tokens / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        bits_per_token = overall_bits_done / total_tokens if total_tokens > 0 else 0.0
        progress_callback(
            ProgressSnapshot(
                phase="decode",
                segment_name=segment_name,
                segment_tokens=segment_tokens,
                total_tokens=total_tokens,
                token_budget=max_tokens,
                segment_bits_done=decoder.resolved_bits,
                segment_bits_total=decoder.total_bits,
                overall_bits_done=overall_bits_done,
                overall_bits_total=overall_bits_total,
                elapsed_seconds=elapsed_seconds,
                tokens_per_second=tokens_per_second,
                bits_per_token=bits_per_token,
                finished=decoder.finished,
            )
        )
