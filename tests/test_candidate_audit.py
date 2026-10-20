import unittest
import numpy as np
from ghostext.candidate_policy import audit_candidate_selection, select_candidates
from ghostext.config import CandidatePolicyConfig
from ghostext.model_backend import RawNextTokenDistribution, TokenProb
from ghostext.quantization import quantize_candidates


class CandidateAuditTests(unittest.TestCase):
    def test_removed_mass_and_selection_agreement(self):
        class Backend:
            def render(self, token_ids):
                return "".join(str(x) for x in token_ids)
            def tokenize(self, text, prompt):
                # Token 1 cannot survive detokenize/re-tokenize.
                return [int(c) if c != "1" else 99 for c in text]
            def token_text(self, token_id):
                return str(token_id)
        backend = Backend()
        policy = CandidatePolicyConfig(top_p=1, max_candidates=2, min_entropy_bits=0)
        probs = [TokenProb(str(i), i, p) for i, p in enumerate([.6, .3, .1])]
        raw = RawNextTokenDistribution(np.arange(3), np.log([.6, .3, .1]))
        for distribution in [probs, raw]:
            audit = audit_candidate_selection(
                distribution, policy, backend=backend, prompt="test", generated_token_ids=[]
            )
            selected = select_candidates(
                distribution, policy, backend=backend, prompt="test", generated_token_ids=[]
            )
            self.assertEqual(audit.stable_selection, selected)
            self.assertAlmostEqual(audit.truncated_tail_mass, .1)
            self.assertAlmostEqual(audit.stability_removed_mass, 1 / 3)

    def test_pseudocount_quantizer_exceeds_historical_bound(self):
        probabilities = [1 - 63e-9] + [1e-9] * 63
        policy = CandidatePolicyConfig(top_p=1, max_candidates=64, min_entropy_bits=0)
        selection = select_candidates(
            [TokenProb(str(i), i, p) for i, p in enumerate(probabilities)], policy
        )
        q = quantize_candidates(selection, 4096)
        tv = sum(abs(e.probability - e.frequency / 4096) for e in q.entries) / 2
        self.assertGreater(tv, len(q.entries) / (2 * 4096))
        self.assertLessEqual(tv, len(q.entries) / 4096)
