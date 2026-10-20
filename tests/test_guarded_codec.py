"""Distribution and recovery checks that expose the v1 endpoint defect."""
from collections import Counter
from fractions import Fraction
import unittest

from ghostext.candidate_policy import select_candidates
from ghostext.codec import (
    MessageSegmentDecoder, MessageSegmentEncoder, interval_rounding_tv,
)
from ghostext.config import CandidatePolicyConfig, CodecConfig
from ghostext.model_backend import TokenProb
from ghostext.quantization import quantize_candidates


def quantized(probabilities, total=4096):
    selection = select_candidates(
        [TokenProb(str(i), i, p) for i, p in enumerate(probabilities)],
        CandidatePolicyConfig(top_p=1.0, max_candidates=len(probabilities),
                              min_entropy_bits=0, enforce_retokenization_stability=False),
    )
    return quantize_candidates(selection, total)


class GuardedCodecTests(unittest.TestCase):
    def test_legacy_endpoint_distribution_and_exact_tv(self):
        distribution = quantized([1 / 64] * 64)
        endings = Counter()
        for value in range(256):
            encoder = MessageSegmentEncoder(bytes([value]))
            first, _ = encoder.choose(distribution)
            self.assertEqual(encoder.high - encoder.low, 4)
            self.assertEqual(
                interval_rounding_tv(encoder.low, encoder.high, distribution),
                Fraction(15, 16),
            )
            last, _ = encoder.choose(distribution)
            self.assertTrue(encoder.finished)
            endings[last] += 1
            decoder = MessageSegmentDecoder(1)
            decoder.absorb(distribution, first)
            decoder.absorb(distribution, last)
            self.assertEqual(decoder.to_bytes(), bytes([value]))
        self.assertEqual(endings, {15: 64, 31: 64, 47: 64, 63: 64})

    def test_guard_removes_uniform64_endpoint_support_signal_exhaustively(self):
        distribution = quantized([1 / 64] * 64)
        paths = Counter()
        for value in range(256):
            for guard in range(16):
                encoder = MessageSegmentEncoder(bytes([value]), guard_bits=4, guard_value=guard)
                decoder = MessageSegmentDecoder(1, guard_bits=4)
                path = []
                while not encoder.finished:
                    self.assertEqual(
                        interval_rounding_tv(encoder.low, encoder.high, distribution), 0
                    )
                    index, _ = encoder.choose(distribution)
                    path.append(index)
                    decoder.absorb(distribution, index)
                self.assertEqual(len(path), 2)
                self.assertTrue(decoder.finished)
                self.assertEqual(decoder.to_bytes(), bytes([value]))
                paths[tuple(path)] += 1
        self.assertEqual(len(paths), 64**2)
        self.assertEqual(set(paths.values()), {1})

    def test_context_dependent_distributions_exhaustive_recovery(self):
        # Decoder's next distribution depends on the observed path, not target.
        models = [quantized([.61, .28, .11]), quantized([.1, .2, .3, .4])]
        for guard_bits in [0, 3, 128]:
            for value in range(256):
                encoder = MessageSegmentEncoder(bytes([value]), guard_bits=guard_bits)
                decoder = MessageSegmentDecoder(1, guard_bits=guard_bits)
                history = []
                for _ in range(64):
                    if encoder.finished:
                        break
                    model = models[sum(history) % 2]
                    index, _ = encoder.choose(model)
                    decoder.absorb(model, index)
                    history.append(index)
                self.assertTrue(encoder.finished)
                self.assertTrue(decoder.finished)
                self.assertEqual(decoder.to_bytes(), bytes([value]))
                self.assertEqual(encoder.resolved_bits, 8)

    def test_configuration_and_invalid_guard_values(self):
        self.assertEqual(CodecConfig().guard_bits, 128)
        self.assertEqual(CodecConfig(guard_bits=0).as_protocol_dict()["version"], "finite-interval-v1")
        self.assertNotEqual(CodecConfig().as_protocol_dict(), CodecConfig(guard_bits=0).as_protocol_dict())
        for kwargs in [{"guard_bits": -1}, {"guard_bits": 3, "guard_value": 8},
                       {"guard_bits": 3, "guard_value": -1}]:
            with self.assertRaises(ValueError):
                MessageSegmentEncoder(b"a", **kwargs)
        for guard_bits in [0, 128]:
            self.assertTrue(MessageSegmentEncoder(b"", guard_bits=guard_bits).finished)
            self.assertEqual(MessageSegmentDecoder(0, guard_bits=guard_bits).to_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
