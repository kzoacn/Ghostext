import unittest

from hidetext.codec import MessageSegmentDecoder, MessageSegmentEncoder
from hidetext.model_backend import TokenProb
from hidetext.candidate_policy import select_candidates
from hidetext.config import CandidatePolicyConfig
from hidetext.quantization import quantize_candidates


def make_quantized(distribution: list[tuple[str, int, float]]):
    selection = select_candidates(
        [TokenProb(token, token_id, prob) for token, token_id, prob in distribution],
        CandidatePolicyConfig(top_p=1.0, max_candidates=16, min_entropy_bits=0.0),
    )
    return quantize_candidates(selection, 256)


class CodecToyTests(unittest.TestCase):
    def test_segment_roundtrip(self) -> None:
        models = [
            make_quantized([("a", 0, 0.55), ("b", 1, 0.30), ("c", 2, 0.15)]),
            make_quantized([("d", 0, 0.40), ("e", 1, 0.35), ("f", 2, 0.25)]),
            make_quantized([("g", 0, 0.70), ("h", 1, 0.20), ("i", 2, 0.10)]),
            make_quantized([("j", 0, 0.33), ("k", 1, 0.33), ("l", 2, 0.34)]),
            make_quantized([("m", 0, 0.60), ("n", 1, 0.25), ("o", 2, 0.15)]),
            make_quantized([("p", 0, 0.50), ("q", 1, 0.30), ("r", 2, 0.20)]),
            make_quantized([("s", 0, 0.45), ("t", 1, 0.35), ("u", 2, 0.20)]),
            make_quantized([("v", 0, 0.55), ("w", 1, 0.25), ("x", 2, 0.20)]),
        ]

        encoder = MessageSegmentEncoder(b"\x9a")
        symbols: list[tuple[object, int]] = []
        for model in models:
            if encoder.finished:
                break
            index, _ = encoder.choose(model)
            symbols.append((model, index))

        self.assertTrue(encoder.finished)

        decoder = MessageSegmentDecoder(1)
        for model, index in symbols:
            decoder.absorb(model, index)
        self.assertTrue(decoder.finished)
        self.assertEqual(decoder.to_bytes(), b"\x9a")

    def test_empty_segment_finishes_immediately(self) -> None:
        encoder = MessageSegmentEncoder(b"")
        decoder = MessageSegmentDecoder(0)
        self.assertTrue(encoder.finished)
        self.assertTrue(decoder.finished)
        self.assertEqual(decoder.to_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
