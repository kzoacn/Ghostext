from __future__ import annotations

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .config import CryptoConfig, PROTOCOL_VERSION
from .errors import ConfigMismatchError, IntegrityError, PacketError
from .packet import InternalHeader, packet_bootstrap_size, split_packet


KDF_ID_SCRYPT = 1
AEAD_ID_CHACHA20_POLY1305 = 1
DERIVATION_CONTEXT = b"hidetext/packet-v2/keys"
HEADER_AAD = b"hidetext/packet-v2/header"
BODY_AAD_PREFIX = b"hidetext/packet-v2/body"


def derive_key(passphrase: str, salt: bytes, crypto_config: CryptoConfig) -> bytes:
    kdf = Scrypt(
        salt=salt,
        length=32,
        n=crypto_config.kdf_n,
        r=crypto_config.kdf_r,
        p=crypto_config.kdf_p,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def _derive_packet_subkeys(master_key: bytes) -> tuple[bytes, bytes]:
    material = HKDF(
        algorithm=hashes.SHA256(),
        length=64,
        salt=None,
        info=DERIVATION_CONTEXT,
    ).derive(master_key)
    return material[:32], material[32:]


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
    body_nonce = nonce or os.urandom(crypto_config.nonce_len)
    header_nonce = os.urandom(crypto_config.nonce_len)
    if len(salt) != crypto_config.salt_len:
        raise PacketError("salt length does not match crypto config")
    if len(body_nonce) != crypto_config.nonce_len:
        raise PacketError("nonce length does not match crypto config")

    master_key = derive_key(passphrase, salt, crypto_config)
    header_key, body_key = _derive_packet_subkeys(master_key)

    body_ciphertext = ChaCha20Poly1305(body_key).encrypt(
        body_nonce,
        plaintext,
        BODY_AAD_PREFIX + salt + body_nonce,
    )
    internal_header = InternalHeader.build(
        body_ciphertext_len=len(body_ciphertext),
        config_fingerprint=config_fingerprint,
        kdf_id=KDF_ID_SCRYPT,
        aead_id=AEAD_ID_CHACHA20_POLY1305,
    )
    sealed_header = ChaCha20Poly1305(header_key).encrypt(
        header_nonce,
        internal_header.pack(),
        HEADER_AAD + salt + header_nonce,
    )
    return salt + header_nonce + sealed_header + body_nonce + body_ciphertext


def decrypt_bootstrap_header(
    bootstrap: bytes,
    *,
    passphrase: str,
    crypto_config: CryptoConfig,
) -> InternalHeader:
    salt_len = crypto_config.salt_len
    nonce_len = crypto_config.nonce_len
    expected_bootstrap_len = packet_bootstrap_size(salt_len, nonce_len)
    if len(bootstrap) != expected_bootstrap_len:
        raise PacketError("opaque bootstrap length mismatch")

    cursor = 0
    salt = bootstrap[cursor : cursor + salt_len]
    cursor += salt_len
    header_nonce = bootstrap[cursor : cursor + nonce_len]
    cursor += nonce_len
    sealed_header = bootstrap[cursor:]

    master_key = derive_key(passphrase, salt, crypto_config)
    header_key, _ = _derive_packet_subkeys(master_key)
    try:
        internal_header_bytes = ChaCha20Poly1305(header_key).decrypt(
            header_nonce,
            sealed_header,
            HEADER_AAD + salt + header_nonce,
        )
    except InvalidTag as exc:
        raise IntegrityError("authenticated header decryption failed") from exc

    internal_header = InternalHeader.unpack(internal_header_bytes)
    if internal_header.version != PROTOCOL_VERSION:
        raise ConfigMismatchError("protocol version mismatch")
    if internal_header.kdf_id != KDF_ID_SCRYPT or internal_header.aead_id != AEAD_ID_CHACHA20_POLY1305:
        raise ConfigMismatchError("crypto algorithm id mismatch")
    return internal_header


def decrypt_packet(
    packet: bytes,
    *,
    passphrase: str,
    expected_config_fingerprint: int,
    crypto_config: CryptoConfig,
) -> bytes:
    bootstrap, body_ciphertext = split_packet(
        packet,
        salt_len=crypto_config.salt_len,
        nonce_len=crypto_config.nonce_len,
    )
    salt_len = crypto_config.salt_len
    nonce_len = crypto_config.nonce_len

    cursor = 0
    salt = bootstrap[cursor : cursor + salt_len]
    internal_header = decrypt_bootstrap_header(
        bootstrap,
        passphrase=passphrase,
        crypto_config=crypto_config,
    )
    if internal_header.config_fingerprint != expected_config_fingerprint:
        raise ConfigMismatchError("config fingerprint mismatch")

    master_key = derive_key(passphrase, salt, crypto_config)
    _, body_key = _derive_packet_subkeys(master_key)
    if len(body_ciphertext) < nonce_len:
        raise PacketError("packet missing body nonce")
    body_nonce = body_ciphertext[:nonce_len]
    encrypted_body = body_ciphertext[nonce_len:]
    if len(encrypted_body) != internal_header.body_ciphertext_len:
        raise PacketError("body ciphertext length mismatch")

    try:
        return ChaCha20Poly1305(body_key).decrypt(
            body_nonce,
            encrypted_body,
            BODY_AAD_PREFIX + salt + body_nonce,
        )
    except InvalidTag as exc:
        raise IntegrityError("authenticated decryption failed") from exc
