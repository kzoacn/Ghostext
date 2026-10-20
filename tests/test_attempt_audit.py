import unittest
from unittest.mock import patch

from ghostext.config import CodecConfig, RuntimeConfig
from ghostext.encoder import StegoEncoder, SegmentStats
from ghostext.errors import UnsafeTokenizationError, EncodingExhaustedError
from ghostext.model_backend import ToyCharBackend


class AttemptAuditTests(unittest.TestCase):
    def test_failed_and_successful_attempts_are_recorded(self):
        encoder = StegoEncoder(ToyCharBackend(), RuntimeConfig(
            codec=CodecConfig(max_encode_attempts=2, natural_tail_max_tokens=0)))
        records = []
        calls = 0

        def fake_encode(*, attempt_tokens, **kwargs):
            nonlocal calls
            calls += 1
            attempt_tokens.append(0)
            if calls == 1:
                raise UnsafeTokenizationError("controlled failure")
            return attempt_tokens, [SegmentStats("body", 1, 1, 8)]

        with patch.object(encoder, "_encode_packet", side_effect=fake_encode):
            encoded = encoder.encode("m", passphrase="public-test", prompt="p",
                                     attempt_callback=records.append,
                                     retry_notice_callback=lambda _: None)
        self.assertEqual(encoded.attempts_used, 2)
        self.assertEqual([r.success for r in records], [False, True])
        self.assertEqual([r.attempt_index for r in records], [1, 2])
        self.assertEqual(records[0].token_ids, (0,))
        self.assertEqual(records[0].failure_class, "UnsafeTokenizationError")
        self.assertNotEqual(records[0].packet_sha256, records[1].packet_sha256)

    def test_terminal_failure_is_also_recorded(self):
        encoder = StegoEncoder(ToyCharBackend())
        records = []
        with patch.object(encoder, "_encode_packet",
                          side_effect=EncodingExhaustedError("controlled budget")):
            with self.assertRaises(EncodingExhaustedError):
                encoder.encode("m", passphrase="public-test", prompt="p",
                               attempt_callback=records.append)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].success)
        self.assertEqual(records[0].failure_class, "EncodingExhaustedError")
