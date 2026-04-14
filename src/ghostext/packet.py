from __future__ import annotations

from dataclasses import dataclass
import struct

from .config import PROTOCOL_VERSION
from .errors import PacketError


AEAD_TAG_LEN = 16
INTERNAL_HEADER_STRUCT = struct.Struct(">BBBBIQ")
INTERNAL_HEADER_SIZE = INTERNAL_HEADER_STRUCT.size


@dataclass(frozen=True)
class InternalHeader:
    version: int
    flags: int
    kdf_id: int
    aead_id: int
    body_ciphertext_len: int
    config_fingerprint: int

    def pack(self) -> bytes:
        return INTERNAL_HEADER_STRUCT.pack(
            self.version,
            self.flags,
            self.kdf_id,
            self.aead_id,
            self.body_ciphertext_len,
            self.config_fingerprint,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "InternalHeader":
        if len(data) != INTERNAL_HEADER_SIZE:
            raise PacketError(f"internal header must be exactly {INTERNAL_HEADER_SIZE} bytes")
        version, flags, kdf_id, aead_id, body_ciphertext_len, config_fingerprint = (
            INTERNAL_HEADER_STRUCT.unpack(data)
        )
        return cls(
            version=version,
            flags=flags,
            kdf_id=kdf_id,
            aead_id=aead_id,
            body_ciphertext_len=body_ciphertext_len,
            config_fingerprint=config_fingerprint,
        )

    @classmethod
    def build(
        cls,
        *,
        body_ciphertext_len: int,
        config_fingerprint: int,
        flags: int = 0,
        kdf_id: int = 1,
        aead_id: int = 1,
    ) -> "InternalHeader":
        return cls(
            version=PROTOCOL_VERSION,
            flags=flags,
            kdf_id=kdf_id,
            aead_id=aead_id,
            body_ciphertext_len=body_ciphertext_len,
            config_fingerprint=config_fingerprint,
        )


def packet_bootstrap_size(salt_len: int, nonce_len: int) -> int:
    if salt_len <= 0 or nonce_len <= 0:
        raise PacketError("salt_len and nonce_len must be positive")
    # bootstrap = salt || header_nonce || sealed_internal_header
    # sealed_internal_header = internal_header || tag
    return salt_len + nonce_len + INTERNAL_HEADER_SIZE + AEAD_TAG_LEN


def split_packet(packet: bytes, *, salt_len: int, nonce_len: int) -> tuple[bytes, bytes]:
    bootstrap_len = packet_bootstrap_size(salt_len, nonce_len)
    if len(packet) < bootstrap_len:
        raise PacketError("packet shorter than opaque bootstrap")
    bootstrap = packet[:bootstrap_len]
    body_ciphertext = packet[bootstrap_len:]
    return bootstrap, body_ciphertext

