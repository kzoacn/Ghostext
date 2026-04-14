import unittest

from ghostext.config import RuntimeConfig
from ghostext.decoder import StegoDecoder
from ghostext.encoder import StegoEncoder
from ghostext.model_backend import ToyCharBackend


class RoundTripEnTests(unittest.TestCase):
    def test_roundtrip_english_message(self) -> None:
        backend = ToyCharBackend()
        config = RuntimeConfig(seed=29)
        prompt = "Write a calm and readable English paragraph."
        message = "Meet me near the station at seven."
        passphrase = "pass-en"

        encoded = StegoEncoder(backend, config).encode(
            message,
            passphrase=passphrase,
            prompt=prompt,
        )
        decoded = StegoDecoder(backend, config).decode(
            encoded.text,
            passphrase=passphrase,
            prompt=prompt,
        )

        self.assertEqual(decoded.plaintext, message)
        self.assertGreater(encoded.total_tokens, 0)
        self.assertTrue(any(ch.isalpha() for ch in encoded.text))


if __name__ == "__main__":
    unittest.main()

