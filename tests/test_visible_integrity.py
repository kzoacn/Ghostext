"""Separate authenticated packet changes from guard-only visible text edits."""
import unittest
from unittest import mock
import numpy as np
from ghostext.config import CandidatePolicyConfig, CodecConfig, RuntimeConfig
from ghostext.decoder import StegoDecoder
from ghostext.encoder import StegoEncoder
from ghostext.errors import IntegrityError
from ghostext.model_backend import BackendMetadata, RawNextTokenDistribution


class Uniform64Backend:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"
    metadata = BackendMetadata(model_id="uniform64", tokenizer_hash="identity64",
                               backend_id="integrity-test")

    def distribution(self, prompt, generated_token_ids, seed):
        return RawNextTokenDistribution(np.arange(64), np.zeros(64))

    def render(self, ids):
        return "".join(self.alphabet[i] for i in ids)

    def tokenize(self, text, prompt):
        return [self.alphabet.index(c) for c in text]

    def token_text(self, token_id):
        return self.alphabet[token_id]


class VisibleIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.backend = Uniform64Backend()
        self.config = RuntimeConfig(
            candidate_policy=CandidatePolicyConfig(top_p=1, max_candidates=64, min_entropy_bits=0),
            codec=CodecConfig(total_frequency=4096, guard_bits=128, natural_tail_max_tokens=0))
        self.message, self.passphrase, self.prompt = "test payload0000", "test-key", "public"
        # A fixed cryptographic test fixture only. Production still uses OS randomness.
        with mock.patch("ghostext.crypto.os.urandom", side_effect=lambda n: b"h" * n), \
                mock.patch("ghostext.codec.secrets.randbits", return_value=0):
            self.encoded = StegoEncoder(self.backend, self.config).encode(
                self.message, passphrase=self.passphrase, prompt=self.prompt,
                salt=b"s" * 16, nonce=b"n" * 12)
        # The 44-byte body has 352 payload bits: its last six-bit symbol contains
        # four packet bits followed by two guard bits.
        self.assertEqual(len(self.encoded.packet) - 60, 44)
        self.assertEqual(self.encoded.token_ids[-1] % 4, 0)
        self.assertEqual(self.encoded.tail_tokens, 0)

    def decode(self, ids):
        return StegoDecoder(self.backend, self.config).decode(
            self.backend.render(ids), passphrase=self.passphrase, prompt=self.prompt)

    def test_guard_only_packet_token_edit_preserves_authenticated_payload(self):
        changed = list(self.encoded.token_ids)
        changed[-1] ^= 1
        result = self.decode(changed)
        self.assertNotEqual(self.backend.render(changed), self.encoded.text)
        self.assertEqual(result.packet, self.encoded.packet)
        self.assertEqual(result.plaintext, self.message)
        self.assertEqual(result.consumed_tokens, self.encoded.packet_tokens)

    def test_packet_changing_terminal_edit_fails_authentication(self):
        changed = list(self.encoded.token_ids)
        changed[-1] ^= 4  # Flip the lowest packet bit, above the two guard bits.
        with self.assertRaises(IntegrityError):
            self.decode(changed)


if __name__ == "__main__":
    unittest.main()
