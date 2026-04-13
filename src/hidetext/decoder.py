from __future__ import annotations

from dataclasses import dataclass

from .codec import MessageSegmentDecoder
from .config import RuntimeConfig
from .crypto import decrypt_packet
from .errors import ConfigMismatchError, PacketError, SynchronizationError
from .model_backend import TextBackend
from .packet import HEADER_SIZE, PacketHeader
from .pipeline import prepare_quantized_distribution


@dataclass(frozen=True)
class DecodeResult:
    plaintext_bytes: bytes
    plaintext: str
    packet: bytes
    tokens: tuple[str, ...]
    header: PacketHeader
    consumed_tokens: int


class StegoDecoder:
    def __init__(self, backend: TextBackend, config: RuntimeConfig | None = None) -> None:
        self.backend = backend
        self.config = config or RuntimeConfig()

    def decode(self, stego_text: str, *, passphrase: str, prompt: str) -> DecodeResult:
        observed_tokens = self.backend.tokenize(stego_text, prompt)
        consumed_tokens: list[str] = []
        cursor = 0

        header_bytes, cursor = self._decode_segment(
            payload_len=HEADER_SIZE,
            observed_tokens=observed_tokens,
            cursor=cursor,
            consumed_tokens=consumed_tokens,
            prompt=prompt,
            max_tokens=self.config.codec.max_header_tokens,
            segment_name="header",
        )
        header = PacketHeader.unpack(header_bytes)
        expected_fp = self.config.config_fingerprint(
            backend_metadata=self.backend.metadata.as_dict(),
            prompt=prompt,
        )
        if header.config_fingerprint != expected_fp:
            raise ConfigMismatchError("packet config fingerprint does not match runtime")

        body_len = header.body_len
        body_bytes, cursor = self._decode_segment(
            payload_len=body_len,
            observed_tokens=observed_tokens,
            cursor=cursor,
            consumed_tokens=consumed_tokens,
            prompt=prompt,
            max_tokens=self.config.codec.max_body_tokens,
            segment_name="body",
        )
        if cursor != len(observed_tokens):
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
            tokens=tuple(observed_tokens),
            header=header,
            consumed_tokens=cursor,
        )

    def _decode_segment(
        self,
        *,
        payload_len: int,
        observed_tokens: list[str],
        cursor: int,
        consumed_tokens: list[str],
        prompt: str,
        max_tokens: int,
        segment_name: str,
    ) -> tuple[bytes, int]:
        decoder = MessageSegmentDecoder(payload_len)
        steps = 0
        while not decoder.finished:
            if steps >= max_tokens:
                raise SynchronizationError(
                    f"{segment_name} segment exceeded token budget {max_tokens}"
                )
            if cursor >= len(observed_tokens):
                raise PacketError(f"stego text ended before {segment_name} segment resolved")

            quantized = prepare_quantized_distribution(
                self.backend,
                prompt=prompt,
                generated_tokens=consumed_tokens,
                config=self.config,
            )
            observed = observed_tokens[cursor]

            if quantized.allows_encoding:
                try:
                    index = quantized.find_index(observed)
                except KeyError as exc:
                    raise SynchronizationError(
                        f"observed token {observed!r} not in {segment_name} candidate set"
                    ) from exc
                try:
                    decoder.absorb(quantized, index)
                except ValueError as exc:
                    raise SynchronizationError(
                        f"{segment_name} interval collapsed while absorbing token {observed!r}"
                    ) from exc
            else:
                if observed != quantized.top.token:
                    raise SynchronizationError(
                        f"expected deterministic token {quantized.top.token!r}, got {observed!r}"
                    )

            consumed_tokens.append(observed)
            cursor += 1
            steps += 1

        return decoder.to_bytes(), cursor
