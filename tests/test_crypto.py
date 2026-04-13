import unittest

from hidetext.config import RuntimeConfig
from hidetext.crypto import build_packet, decrypt_packet
from hidetext.errors import ConfigMismatchError, IntegrityError
from hidetext.model_backend import ToyCharBackend


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


if __name__ == "__main__":
    unittest.main()

