import unittest

from hidetext.config import RuntimeConfig
from hidetext.decoder import StegoDecoder
from hidetext.encoder import StegoEncoder
from hidetext.errors import HideTextError
from hidetext.model_backend import ToyCharBackend


class FailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ToyCharBackend()
        self.config = RuntimeConfig(seed=7)
        self.prompt = "请写一段简短的中文短文。"
        self.passphrase = "hunter2"
        self.message = "把消息藏在文本里。"
        self.encoded = StegoEncoder(self.backend, self.config).encode(
            self.message,
            passphrase=self.passphrase,
            prompt=self.prompt,
        )

    def test_wrong_seed_fails_closed(self) -> None:
        wrong_config = RuntimeConfig(seed=8)
        with self.assertRaises(HideTextError):
            StegoDecoder(self.backend, wrong_config).decode(
                self.encoded.text,
                passphrase=self.passphrase,
                prompt=self.prompt,
            )

    def test_wrong_prompt_fails(self) -> None:
        with self.assertRaises(HideTextError):
            StegoDecoder(self.backend, self.config).decode(
                self.encoded.text,
                passphrase=self.passphrase,
                prompt="Write an English paragraph instead.",
            )

    def test_mutated_text_fails(self) -> None:
        mutated = self.encoded.text[:-1] + (
            "。"
            if self.encoded.text[-1] != "。"
            else "，"
        )
        with self.assertRaises(HideTextError):
            StegoDecoder(self.backend, self.config).decode(
                mutated,
                passphrase=self.passphrase,
                prompt=self.prompt,
            )


if __name__ == "__main__":
    unittest.main()
