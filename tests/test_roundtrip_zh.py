import unittest

from ghostext.config import RuntimeConfig
from ghostext.decoder import StegoDecoder
from ghostext.encoder import StegoEncoder
from ghostext.model_backend import ToyCharBackend


class RoundTripZhTests(unittest.TestCase):
    def test_roundtrip_chinese_message(self) -> None:
        backend = ToyCharBackend()
        config = RuntimeConfig(seed=17)
        prompt = "请写一段温柔自然的中文短文。"
        message = "今晚七点在老地方见。"
        passphrase = "pass-zh"

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
        self.assertGreater(encoded.bits_per_token, 0.0)
        self.assertTrue(any("\u4e00" <= ch <= "\u9fff" for ch in encoded.text))


if __name__ == "__main__":
    unittest.main()

