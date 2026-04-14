import unittest

import numpy as np

from ghostext.candidate_policy import select_candidates
from ghostext.config import CandidatePolicyConfig
from ghostext.errors import ModelBackendError, UnsafeTokenizationError
from ghostext.model_backend import BackendMetadata, RawNextTokenDistribution


class AmbiguousTokenizerBackend:
    def __init__(self) -> None:
        self._tokens = {
            0: "冷",
            1: "却",
            2: "冷却",
            3: "好",
            4: "。",
        }
        self._metadata = BackendMetadata(
            model_id="ambiguous-tokenizer-backend",
            tokenizer_hash="ambiguous-tokenizer-hash",
            backend_id="ambiguous-tokenizer",
        )

    @property
    def metadata(self) -> BackendMetadata:
        return self._metadata

    def tokenize(self, text: str, prompt: str) -> list[int]:
        del prompt
        token_ids: list[int] = []
        cursor = 0
        while cursor < len(text):
            if text.startswith("冷却", cursor):
                token_ids.append(2)
                cursor += 2
                continue
            ch = text[cursor]
            if ch == "冷":
                token_ids.append(0)
            elif ch == "却":
                token_ids.append(1)
            elif ch == "好":
                token_ids.append(3)
            elif ch == "。":
                token_ids.append(4)
            else:
                raise ModelBackendError(f"unsupported character: {ch!r}")
            cursor += 1
        return token_ids

    def render(self, token_ids: list[int]) -> str:
        return "".join(self._tokens[token_id] for token_id in token_ids)

    def token_text(self, token_id: int) -> str:
        return self._tokens[token_id]

    def distribution(
        self,
        prompt: str,
        generated_token_ids: list[int],
        seed: int,
    ) -> RawNextTokenDistribution:
        del prompt, generated_token_ids, seed
        raise NotImplementedError


class TokenizationStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = AmbiguousTokenizerBackend()
        self.config = CandidatePolicyConfig(
            top_p=1.0,
            max_candidates=8,
            min_entropy_bits=0.0,
        )

    def test_guard_filters_token_that_retokenizes_into_different_path(self) -> None:
        selection = select_candidates(
            RawNextTokenDistribution(
                token_ids=np.asarray([1, 3], dtype=np.int32),
                logits=np.asarray([0.0, -0.2], dtype=np.float64),
            ),
            self.config,
            backend=self.backend,
            prompt="中文",
            generated_token_ids=[0],
        )

        self.assertEqual([entry.token_id for entry in selection.entries], [3])
        self.assertFalse(selection.allows_encoding)

    def test_guard_fails_closed_when_no_safe_candidate_remains(self) -> None:
        with self.assertRaises(UnsafeTokenizationError):
            select_candidates(
                RawNextTokenDistribution(
                    token_ids=np.asarray([1], dtype=np.int32),
                    logits=np.asarray([0.0], dtype=np.float64),
                ),
                self.config,
                backend=self.backend,
                prompt="中文",
                generated_token_ids=[0],
            )


if __name__ == "__main__":
    unittest.main()
