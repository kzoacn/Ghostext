import unittest

from hidetext.config import RuntimeConfig
from hidetext.crypto import build_packet, decrypt_bootstrap_header, decrypt_packet
from hidetext.errors import ConfigMismatchError, IntegrityError
from hidetext.model_backend import ToyCharBackend
from hidetext.packet import packet_bootstrap_size


class CryptoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RuntimeConfig()
        self.backend = ToyCharBackend()
        self.prompt = "请写一段安静的中文短文。"
        self.fingerprint = self.config.config_fingerprint(
            backend_metadata=self.backend.metadata.as_dict(),
            prompt=self.prompt,
        )

    def test_encrypt_decrypt_roundtrip(self) -> None:
        packet = build_packet(
            b"secret message",
            passphrase="hunter2",
            config_fingerprint=self.fingerprint,
            crypto_config=self.config.crypto,
            salt=b"s" * self.config.crypto.salt_len,
            nonce=b"n" * self.config.crypto.nonce_len,
        )
        plaintext = decrypt_packet(
            packet,
            passphrase="hunter2",
            expected_config_fingerprint=self.fingerprint,
            crypto_config=self.config.crypto,
        )
        self.assertEqual(plaintext, b"secret message")

    def test_config_mismatch_raises(self) -> None:
        packet = build_packet(
            b"secret",
            passphrase="hunter2",
            config_fingerprint=self.fingerprint,
            crypto_config=self.config.crypto,
            salt=b"s" * self.config.crypto.salt_len,
            nonce=b"n" * self.config.crypto.nonce_len,
        )
        with self.assertRaises(ConfigMismatchError):
            decrypt_packet(
                packet,
                passphrase="hunter2",
                expected_config_fingerprint=self.fingerprint + 1,
                crypto_config=self.config.crypto,
            )

    def test_invalid_passphrase_raises_integrity_error(self) -> None:
        packet = build_packet(
            b"secret",
            passphrase="hunter2",
            config_fingerprint=self.fingerprint,
            crypto_config=self.config.crypto,
            salt=b"s" * self.config.crypto.salt_len,
            nonce=b"n" * self.config.crypto.nonce_len,
        )
        with self.assertRaises(IntegrityError):
            decrypt_packet(
                packet,
                passphrase="wrong",
                expected_config_fingerprint=self.fingerprint,
                crypto_config=self.config.crypto,
            )

    def test_bootstrap_header_is_not_plaintext(self) -> None:
        packet = build_packet(
            b"secret message",
            passphrase="hunter2",
            config_fingerprint=self.fingerprint,
            crypto_config=self.config.crypto,
            salt=b"s" * self.config.crypto.salt_len,
            nonce=b"n" * self.config.crypto.nonce_len,
        )
        bootstrap_len = packet_bootstrap_size(
            self.config.crypto.salt_len,
            self.config.crypto.nonce_len,
        )
        bootstrap = packet[:bootstrap_len]
        self.assertNotIn(b"HDTX", bootstrap)
        self.assertNotEqual(bootstrap[:4], b"HDTX")

    def test_decrypt_bootstrap_header_roundtrip(self) -> None:
        packet = build_packet(
            b"secret message",
            passphrase="hunter2",
            config_fingerprint=self.fingerprint,
            crypto_config=self.config.crypto,
            salt=b"s" * self.config.crypto.salt_len,
            nonce=b"n" * self.config.crypto.nonce_len,
        )
        bootstrap_len = packet_bootstrap_size(
            self.config.crypto.salt_len,
            self.config.crypto.nonce_len,
        )
        header = decrypt_bootstrap_header(
            packet[:bootstrap_len],
            passphrase="hunter2",
            crypto_config=self.config.crypto,
        )
        self.assertEqual(header.config_fingerprint, self.fingerprint)
        self.assertGreater(header.body_ciphertext_len, 0)


if __name__ == "__main__":
    unittest.main()
