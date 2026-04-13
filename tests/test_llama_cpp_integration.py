import os
import unittest

from hidetext.config import CandidatePolicyConfig, CodecConfig, RuntimeConfig
from hidetext.decoder import StegoDecoder
from hidetext.encoder import StegoEncoder
from hidetext.llama_cpp_backend import LlamaCppBackendConfig, QwenLlamaCppBackend


MODEL_PATH = os.environ.get("HIDETEXT_LLAMA_MODEL_PATH")


@unittest.skipUnless(MODEL_PATH, "set HIDETEXT_LLAMA_MODEL_PATH to run real-model integration")
class LlamaCppIntegrationTests(unittest.TestCase):
    def test_qwen_roundtrip_smoke(self) -> None:
        assert MODEL_PATH is not None
        seed = int(os.environ.get("HIDETEXT_LLAMA_SEED", "7"))
        backend = QwenLlamaCppBackend(
            LlamaCppBackendConfig(
                model_path=MODEL_PATH,
                n_ctx=int(os.environ.get("HIDETEXT_LLAMA_CTX", "4096")),
                n_batch=int(os.environ.get("HIDETEXT_LLAMA_BATCH", "256")),
                n_threads=int(os.environ.get("HIDETEXT_LLAMA_THREADS", "8")),
                seed=seed,
            )
        )
        config = RuntimeConfig(
            seed=seed,
            candidate_policy=CandidatePolicyConfig(
                top_p=0.995,
                max_candidates=64,
                min_entropy_bits=0.0,
            ),
            codec=CodecConfig(
                total_frequency=4096,
                max_header_tokens=1024,
                max_body_tokens=3072,
            ),
        )
        prompt = "请写一段自然、简短、连贯的中文段落，描写傍晚散步时看到的街景。"
        message = "真实模型联调成功"
        passphrase = "qwen-real-test"

        encoded = StegoEncoder(backend, config).encode(
            message,
            passphrase=passphrase,
            prompt=prompt,
            salt=b"s" * config.crypto.salt_len,
            nonce=b"n" * config.crypto.nonce_len,
        )
        decoded = StegoDecoder(backend, config).decode(
            encoded.text,
            passphrase=passphrase,
            prompt=prompt,
        )
        self.assertEqual(decoded.plaintext, message)
        self.assertGreater(encoded.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
