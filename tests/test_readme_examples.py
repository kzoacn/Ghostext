import unittest

from ghostext.config import RuntimeConfig
from ghostext.decoder import StegoDecoder
from ghostext.encoder import StegoEncoder
from ghostext.model_backend import ToyCharBackend


EXAMPLES = [
    ("写一首与河流有关的长诗", "七点河边见"),
    ("虚构一个北欧神话的神明", "目标已就位"),
    ("想象地球的第二个卫星", "核弹即将发射"),
    ("发明一种海洋生物的货币", "股票会上涨"),
    ("设计一个与做梦有关的游戏", "目标正在睡觉"),
]


class ReadmeExampleTests(unittest.TestCase):
    def test_curated_examples_roundtrip(self) -> None:
        backend = ToyCharBackend()
        for index, (prompt, message) in enumerate(EXAMPLES, start=1):
            with self.subTest(prompt=prompt, message=message):
                config = RuntimeConfig(seed=100 + index)
                passphrase = f"example-pass-{index}"
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
                self.assertGreaterEqual(encoded.tail_tokens, 0)


if __name__ == "__main__":
    unittest.main()
