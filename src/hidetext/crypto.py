from __future__ import annotations

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .config import CryptoConfig
from .errors import ConfigMismatchError, IntegrityError, PacketError
from .packet import PacketHeader, split_packet


def derive_key(passphrase: str, salt: bytes, crypto_config: CryptoConfig) -> bytes:
    kdf = Scrypt(
        salt=salt,
        length=32,
        n=crypto_config.kdf_n,
        r=crypto_config.kdf_r,
        p=crypto_config.kdf_p,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def build_packet(
    plaintext: bytes,
    *,
    passphrase: str,
    config_fingerprint: int,
    crypto_config: CryptoConfig,
    salt: bytes | None = None,
    nonce: bytes | None = None,
) -> bytes:
    salt = salt or os.urandom(crypto_config.salt_len)
    nonce = nonce or os.urandom(crypto_config.nonce_len)
    if len(salt) != crypto_config.salt_len:
        raise PacketError("salt length does not match crypto config")
    if len(nonce) != crypto_config.nonce_len:
        raise PacketError("nonce length does not match crypto config")

    key = derive_key(passphrase, salt, crypto_config)
    ciphertext_len = len(plaintext) + 16
    header = PacketHeader.build(
        salt_len=len(salt),
        nonce_len=len(nonce),
        ciphertext_len=ciphertext_len,
        config_fingerprint=config_fingerprint,
    )
    aad = header.pack()
    ciphertext = ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)
    return aad + salt + nonce + ciphertext


def decrypt_packet(
    packet: bytes,
    *,
    passphrase: str,
    expected_config_fingerprint: int,
    crypto_config: CryptoConfig,
) -> bytes:
    header, body = split_packet(packet)
    if header.config_fingerprint != expected_config_fingerprint:
        raise ConfigMismatchError("config fingerprint mismatch")
    if header.salt_len != crypto_config.salt_len or header.nonce_len != crypto_config.nonce_len:
        raise ConfigMismatchError("crypto sizing mismatch")

    cursor = 0
    salt = body[cursor : cursor + header.salt_len]
    cursor += header.salt_len
    nonce = body[cursor : cursor + header.nonce_len]
    cursor += header.nonce_len
    ciphertext = body[cursor:]
    if len(ciphertext) != header.ciphertext_len:
        raise PacketError("ciphertext length mismatch")

    key = derive_key(passphrase, salt, crypto_config)
    try:
        return ChaCha20Poly1305(key).decrypt(nonce, ciphertext, header.pack())
    except InvalidTag as exc:
        raise IntegrityError("authenticated decryption failed") from exc

