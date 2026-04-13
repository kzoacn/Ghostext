import unittest

import numpy as np

from hidetext.config import CodecConfig, RuntimeConfig
from hidetext.decoder import StegoDecoder
from hidetext.encoder import StegoEncoder
from hidetext.errors import HideTextError, StallDetectedError
from hidetext.model_backend import BackendMetadata, RawNextTokenDistribution, ToyCharBackend


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

    def test_stall_detector_fails_closed(self) -> None:
        class StallingBackend:
            def __init__(self) -> None:
                self._metadata = BackendMetadata(
                    model_id="stalling-backend",
                    tokenizer_hash="stalling-hash",
                    backend_id="stalling",
                )

            @property
            def metadata(self) -> BackendMetadata:
                return self._metadata

            def tokenize(self, text: str, prompt: str) -> list[int]:
                del prompt
                return [0 for _ in text]

            def render(self, token_ids: list[int]) -> str:
                return "a" * len(token_ids)

            def token_text(self, token_id: int) -> str:
                return "a" if token_id == 0 else "b"

            def distribution(
                self,
                prompt: str,
                generated_token_ids: list[int],
                seed: int,
            ) -> RawNextTokenDistribution:
                del prompt, seed
                if generated_token_ids:
                    return RawNextTokenDistribution(
                        token_ids=np.asarray([0], dtype=np.int32),
                        logits=np.asarray([0.0], dtype=np.float64),
                    )
                return RawNextTokenDistribution(
                    token_ids=np.asarray([0, 1], dtype=np.int32),
                    logits=np.asarray([0.0, 0.0], dtype=np.float64),
                )

        config = RuntimeConfig(
            seed=7,
            codec=CodecConfig(
                total_frequency=256,
                max_header_tokens=100,
                max_body_tokens=100,
                stall_patience_tokens=8,
            ),
        )
        backend = StallingBackend()
        with self.assertRaises(StallDetectedError):
            StegoEncoder(backend, config).encode(
                "stall me",
                passphrase="hunter2",
                prompt="test prompt",
                salt=b"s" * config.crypto.salt_len,
                nonce=b"n" * config.crypto.nonce_len,
            )

    def test_stall_patience_does_not_affect_decode_compatibility(self) -> None:
        encoded_config = RuntimeConfig(
            seed=7,
            codec=CodecConfig(stall_patience_tokens=0),
        )
        encoded = StegoEncoder(self.backend, encoded_config).encode(
            self.message,
            passphrase=self.passphrase,
            prompt=self.prompt,
        )
        decode_config = RuntimeConfig(
            seed=7,
            codec=CodecConfig(stall_patience_tokens=512),
        )
        decoded = StegoDecoder(self.backend, decode_config).decode(
            encoded.text,
            passphrase=self.passphrase,
            prompt=self.prompt,
        )
        self.assertEqual(decoded.plaintext, self.message)


if __name__ == "__main__":
    unittest.main()
