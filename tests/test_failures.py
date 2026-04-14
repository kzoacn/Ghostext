import unittest
from unittest import mock

import numpy as np

from hidetext.config import CandidatePolicyConfig, CodecConfig, RuntimeConfig
from hidetext.decoder import StegoDecoder
from hidetext.encoder import StegoEncoder
from hidetext.errors import HideTextError, LowEntropyRetryLimitError, StallDetectedError
from hidetext.model_backend import BackendMetadata, RawNextTokenDistribution, ToyCharBackend
from hidetext.packet import HEADER_SIZE


class RetrySensitiveBackend:
    def __init__(self) -> None:
        self._metadata = BackendMetadata(
            model_id="retry-sensitive-backend",
            tokenizer_hash="retry-sensitive-hash",
            backend_id="retry-sensitive",
        )
        self._header_tokens = HEADER_SIZE * 8

    @property
    def metadata(self) -> BackendMetadata:
        return self._metadata

    def tokenize(self, text: str, prompt: str) -> list[int]:
        del prompt
        return [0 if ch == "a" else 1 for ch in text]

    def render(self, token_ids: list[int]) -> str:
        return "".join("a" if token_id == 0 else "b" for token_id in token_ids)

    def token_text(self, token_id: int) -> str:
        return "a" if token_id == 0 else "b"

    def distribution(
        self,
        prompt: str,
        generated_token_ids: list[int],
        seed: int,
    ) -> RawNextTokenDistribution:
        del prompt, seed
        if len(generated_token_ids) < self._header_tokens:
            return RawNextTokenDistribution(
                token_ids=np.asarray([0, 1], dtype=np.int32),
                logits=np.asarray([0.0, 0.0], dtype=np.float64),
            )
        if len(generated_token_ids) == self._header_tokens:
            return RawNextTokenDistribution(
                token_ids=np.asarray([0, 1], dtype=np.int32),
                logits=np.asarray([0.0, 0.0], dtype=np.float64),
            )
        first_body_token_id = generated_token_ids[self._header_tokens]
        if first_body_token_id == 0:
            return RawNextTokenDistribution(
                token_ids=np.asarray([0], dtype=np.int32),
                logits=np.asarray([0.0], dtype=np.float64),
            )
        return RawNextTokenDistribution(
            token_ids=np.asarray([0, 1], dtype=np.int32),
            logits=np.asarray([0.0, 0.0], dtype=np.float64),
        )


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
        packet_text = self.encoded.text[: self.encoded.packet_tokens]
        mutated_packet_text = packet_text[:-1] + (
            "。"
            if packet_text[-1] != "。"
            else "，"
        )
        mutated = mutated_packet_text + self.encoded.text[self.encoded.packet_tokens :]
        with self.assertRaises(HideTextError):
            StegoDecoder(self.backend, self.config).decode(
                mutated,
                passphrase=self.passphrase,
                prompt=self.prompt,
            )

    def test_decoder_ignores_trailing_tail_tokens(self) -> None:
        decoded = StegoDecoder(self.backend, self.config).decode(
            self.encoded.text + "。",
            passphrase=self.passphrase,
            prompt=self.prompt,
        )
        self.assertEqual(decoded.plaintext, self.message)
        self.assertGreater(decoded.trailing_tokens, 0)

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

    def test_low_entropy_attempt_retries_with_fresh_packet(self) -> None:
        backend = RetrySensitiveBackend()
        config = RuntimeConfig(
            seed=7,
            candidate_policy=CandidatePolicyConfig(
                top_p=1.0,
                max_candidates=2,
                min_entropy_bits=0.0,
            ),
            codec=CodecConfig(
                total_frequency=2,
                max_header_tokens=HEADER_SIZE * 8,
                max_body_tokens=32,
                stall_patience_tokens=0,
                low_entropy_window_tokens=4,
                low_entropy_threshold_bits=0.1,
                max_encode_attempts=3,
            ),
        )
        first_packet = (b"H" * HEADER_SIZE) + b"\x00"
        second_packet = (b"H" * HEADER_SIZE) + b"\x80"
        with mock.patch(
            "hidetext.encoder.build_packet",
            side_effect=[first_packet, second_packet],
        ) as build_packet_mock:
            result = StegoEncoder(backend, config).encode(
                "retry me",
                passphrase="hunter2",
                prompt="test prompt",
            )

        self.assertEqual(build_packet_mock.call_count, 2)
        self.assertEqual(result.attempts_used, 2)
        self.assertGreater(result.total_tokens, HEADER_SIZE * 8)

    def test_low_entropy_retry_limit_reports_actionable_error(self) -> None:
        backend = RetrySensitiveBackend()
        config = RuntimeConfig(
            seed=7,
            candidate_policy=CandidatePolicyConfig(
                top_p=1.0,
                max_candidates=2,
                min_entropy_bits=0.0,
            ),
            codec=CodecConfig(
                total_frequency=2,
                max_header_tokens=HEADER_SIZE * 8,
                max_body_tokens=32,
                stall_patience_tokens=0,
                low_entropy_window_tokens=4,
                low_entropy_threshold_bits=0.1,
                max_encode_attempts=3,
            ),
        )
        failing_packet = (b"H" * HEADER_SIZE) + b"\x00"
        with mock.patch(
            "hidetext.encoder.build_packet",
            side_effect=[failing_packet, failing_packet, failing_packet],
        ) as build_packet_mock:
            with self.assertRaises(LowEntropyRetryLimitError) as ctx:
                StegoEncoder(backend, config).encode(
                    "retry me",
                    passphrase="hunter2",
                    prompt="test prompt",
                )

        self.assertEqual(build_packet_mock.call_count, 3)
        self.assertIn("Try a different prompt or reduce the message length.", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
