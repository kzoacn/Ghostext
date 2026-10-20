import unittest
import numpy as np
from ghostext.candidate_policy import audit_candidate_selection, select_candidates
from ghostext.config import CandidatePolicyConfig
from ghostext.model_backend import RawNextTokenDistribution, TokenProb
from ghostext.quantization import quantize_candidates
from ghostext.errors import UnsafeTokenizationError


class CandidateAuditTests(unittest.TestCase):
    def test_optional_empty_set_search_is_bounded_and_audited(self):
        class Backend:
            def render(self, ids):
                return "".join(map(str, ids))
            def tokenize(self, text, prompt):
                return [int(c) if int(c) >= 2 else 99 for c in text]
            def token_text(self, token_id):
                return str(token_id)
        backend = Backend()
        probs = [TokenProb(str(i), i, p) for i, p in enumerate([.9, .06, .03, .01])]
        base = CandidatePolicyConfig(top_p=.95, max_candidates=2, min_entropy_bits=0)
        options = dict(backend=backend, prompt="public", generated_token_ids=[])
        with self.assertRaises(UnsafeTokenizationError):
            select_candidates(probs, base, **options)
        expanded = CandidatePolicyConfig(top_p=.95, max_candidates=2, min_entropy_bits=0,
                                         empty_candidate_search=4)
        audit = audit_candidate_selection(probs, expanded, **options)
        self.assertTrue(audit.fallback_used)
        self.assertEqual([r.token_id for r in audit.stable_selection.entries], [2, 3])
        self.assertAlmostEqual(audit.stability_removed_mass, .96)
        self.assertEqual(select_candidates(probs, expanded, **options), audit.stable_selection)
        raw = RawNextTokenDistribution(np.arange(4), np.log([.9, .06, .03, .01]))
        raw_audit = audit_candidate_selection(raw, expanded, **options)
        self.assertTrue(raw_audit.fallback_used)
        self.assertEqual([r.token_id for r in raw_audit.stable_selection.entries], [2, 3])
        self.assertAlmostEqual(raw_audit.stability_removed_mass, .96)
        self.assertNotEqual(base.as_protocol_dict(), expanded.as_protocol_dict())
        self.assertNotIn("empty_candidate_search", base.as_protocol_dict())

    def test_expanded_pool_is_bounded_and_stable_output_keeps_original_cap(self):
        class Backend:
            def render(self, ids): return "".join(map(str, ids))
            def tokenize(self, text, prompt):
                return [int(c) if int(c) >= 3 else 99 for c in text]
            def token_text(self, token_id): return str(token_id)
        backend = Backend()
        raw = RawNextTokenDistribution(np.arange(6), np.log([.8, .1, .06, .02, .015, .005]))
        options = dict(backend=backend, prompt="p", generated_token_ids=[])
        bounded = CandidatePolicyConfig(top_p=1, max_candidates=2, min_entropy_bits=0,
                                        empty_candidate_search=3)
        with self.assertRaises(UnsafeTokenizationError):
            select_candidates(raw, bounded, **options)
        expanded = CandidatePolicyConfig(top_p=1, max_candidates=2, min_entropy_bits=0,
                                         empty_candidate_search=6)
        selected = select_candidates(raw, expanded, **options)
        self.assertEqual([r.token_id for r in selected.entries], [3, 4])
        self.assertAlmostEqual(sum(r.probability for r in selected.entries), 1)
        with self.assertRaises(ValueError):
            CandidatePolicyConfig(max_candidates=64, empty_candidate_search=32)

    def test_search_does_not_change_nonempty_original_selection(self):
        probs = [TokenProb("a", 0, .8), TokenProb("b", 1, .2)]
        class Backend:
            def render(self, ids): return "".join("ab"[i] for i in ids)
            def tokenize(self, text, prompt): return ["ab".index(c) for c in text]
        backend = Backend()
        options = dict(backend=backend, prompt="p", generated_token_ids=[])
        base = CandidatePolicyConfig(top_p=1, max_candidates=2, min_entropy_bits=0)
        expanded = CandidatePolicyConfig(top_p=1, max_candidates=2, min_entropy_bits=0,
                                         empty_candidate_search=4)
        self.assertEqual(select_candidates(probs, base, **options),
                         select_candidates(probs, expanded, **options))
        self.assertFalse(audit_candidate_selection(probs, expanded, **options).fallback_used)

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
