from __future__ import annotations

from dataclasses import dataclass
import struct

from .config import PROTOCOL_VERSION
from .errors import MagicMismatchError, PacketError


PACKET_MAGIC = b"HDTX"
KDF_ID_SCRYPT = 1
AEAD_ID_CHACHA20_POLY1305 = 1
HEADER_STRUCT = struct.Struct(">4sBBBBBB2xIQ")
HEADER_SIZE = HEADER_STRUCT.size


@dataclass(frozen=True)
class PacketHeader:
    version: int
    flags: int
    kdf_id: int
    aead_id: int
    salt_len: int
    nonce_len: int
    ciphertext_len: int
    config_fingerprint: int

    def pack(self) -> bytes:
        return HEADER_STRUCT.pack(
            PACKET_MAGIC,
            self.version,
            self.flags,
            self.kdf_id,
            self.aead_id,
            self.salt_len,
            self.nonce_len,
            self.ciphertext_len,
            self.config_fingerprint,
        )

    @property
    def body_len(self) -> int:
        return self.salt_len + self.nonce_len + self.ciphertext_len

    @property
    def packet_len(self) -> int:
        return HEADER_SIZE + self.body_len

    @classmethod
    def build(
        cls,
        *,
        salt_len: int,
        nonce_len: int,
        ciphertext_len: int,
        config_fingerprint: int,
        flags: int = 0,
    ) -> "PacketHeader":
        return cls(
            version=PROTOCOL_VERSION,
            flags=flags,
            kdf_id=KDF_ID_SCRYPT,
            aead_id=AEAD_ID_CHACHA20_POLY1305,
            salt_len=salt_len,
            nonce_len=nonce_len,
            ciphertext_len=ciphertext_len,
            config_fingerprint=config_fingerprint,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "PacketHeader":
        if len(data) != HEADER_SIZE:
            raise PacketError(f"header must be exactly {HEADER_SIZE} bytes")
        magic, version, flags, kdf_id, aead_id, salt_len, nonce_len, ciphertext_len, config_fp = (
            HEADER_STRUCT.unpack(data)
        )
        if magic != PACKET_MAGIC:
            raise MagicMismatchError("packet magic mismatch")
        if salt_len < 1 or nonce_len < 1:
            raise PacketError("salt_len and nonce_len must be positive")
        return cls(
            version=version,
            flags=flags,
            kdf_id=kdf_id,
            aead_id=aead_id,
            salt_len=salt_len,
            nonce_len=nonce_len,
            ciphertext_len=ciphertext_len,
            config_fingerprint=config_fp,
        )


def split_packet(packet: bytes) -> tuple[PacketHeader, bytes]:
    if len(packet) < HEADER_SIZE:
        raise PacketError("packet shorter than header")
    header = PacketHeader.unpack(packet[:HEADER_SIZE])
    body = packet[HEADER_SIZE:]
    if len(body) != header.body_len:
        raise PacketError("packet body length does not match header")
    return header, body

