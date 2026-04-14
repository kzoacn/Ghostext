import unittest

from ghostext.candidate_policy import select_candidates
from ghostext.config import CandidatePolicyConfig
from ghostext.model_backend import TokenProb
from ghostext.quantization import quantize_candidates


class QuantizationTests(unittest.TestCase):
    def test_quantization_preserves_total_frequency(self) -> None:
        distribution = [
            TokenProb("a", 0, 0.5),
            TokenProb("b", 1, 0.3),
            TokenProb("c", 2, 0.2),
        ]
        selection = select_candidates(
            distribution,
            CandidatePolicyConfig(top_p=1.0, max_candidates=8, min_entropy_bits=0.0),
        )
        quantized = quantize_candidates(selection, 257)
        self.assertEqual(sum(entry.frequency for entry in quantized.entries), 257)
        self.assertTrue(all(entry.frequency >= 1 for entry in quantized.entries))

    def test_quantization_is_deterministic_under_ties(self) -> None:
        distribution = [
            TokenProb("x", 5, 0.25),
            TokenProb("y", 3, 0.25),
            TokenProb("z", 7, 0.25),
            TokenProb("w", 1, 0.25),
        ]
        selection = select_candidates(
            distribution,
            CandidatePolicyConfig(top_p=1.0, max_candidates=8, min_entropy_bits=0.0),
        )
        first = quantize_candidates(selection, 31)
        second = quantize_candidates(selection, 31)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

