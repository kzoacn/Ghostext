from __future__ import annotations

from dataclasses import dataclass

from .codec import MessageSegmentEncoder
from .config import RuntimeConfig
from .crypto import build_packet
from .errors import EncodingExhaustedError
from .model_backend import TextBackend
from .packet import HEADER_SIZE
from .pipeline import prepare_quantized_distribution


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

    @property
    def total_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def bits_per_token(self) -> float:
        if not self.token_ids:
            return 0.0
        return (len(self.packet) * 8) / len(self.token_ids)


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
    ) -> EncodeResult:
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

        header_stats = self._encode_segment(
            segment_name="header",
            payload=packet[:HEADER_SIZE],
            prompt=prompt,
            generated_token_ids=token_ids,
            max_tokens=self.config.codec.max_header_tokens,
        )
        stats.append(header_stats)

        body_stats = self._encode_segment(
            segment_name="body",
            payload=packet[HEADER_SIZE:],
            prompt=prompt,
            generated_token_ids=token_ids,
            max_tokens=self.config.codec.max_body_tokens,
        )
        stats.append(body_stats)

        return EncodeResult(
            text=self.backend.render(token_ids),
            token_ids=tuple(token_ids),
            packet=packet,
            config_fingerprint=config_fingerprint,
            segment_stats=tuple(stats),
        )

    def _encode_segment(
        self,
        *,
        segment_name: str,
        payload: bytes,
        prompt: str,
        generated_token_ids: list[int],
        max_tokens: int,
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

        return SegmentStats(
            name=segment_name,
            tokens_used=steps,
            encoding_steps=encoding_steps,
            embedded_bits=embedded_bits,
        )
