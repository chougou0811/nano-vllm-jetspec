"""CPU-only guards for strict Flash qualification policy and numerical gates."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
import jetspec_flash_qualification as qualification


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.idle = True
        self._jetspec_scheduler = SimpleNamespace(runtime=SimpleNamespace(_attention_backend="sdpa", _prefill_attention_backend="sdpa"))

    def is_finished(self):
        return self.idle

    def configure_jetspec(self, draft, **kwargs):
        self.calls.append((draft, kwargs))
        self._jetspec_scheduler.runtime._attention_backend = kwargs["attention_backend"]


class FlashQualificationTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.args = SimpleNamespace(draft="draft-path", max_model_len=4096)

    def test_threshold_is_fixed_and_rejects_large_error(self):
        expected = torch.ones(16)
        passing = qualification.metrics(expected + .01, expected)
        failed = qualification.metrics(expected + .02, expected)
        self.assertEqual(passing["fixed_scaled_max_and_relative_rms_bound"], 2 ** -6)
        self.assertTrue(passing["passed"])
        self.assertFalse(failed["passed"])

    def test_zero_reference_does_not_hide_relative_error(self):
        self.assertTrue(qualification.metrics(torch.zeros(8), torch.zeros(8))["passed"])
        self.assertFalse(qualification.metrics(torch.ones(8) * 1e-5, torch.zeros(8))["passed"])

    def test_nonfinite_diagnostic_cannot_pass(self):
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            qualification.metrics(torch.tensor([float("nan")]), torch.ones(1))

    def test_explicit_flash_serving_policy_no_chunk(self):
        qualification.configure(self.engine, self.args, 4)
        draft, flags = self.engine.calls[-1]
        self.assertEqual(draft, self.args.draft)
        self.assertEqual(flags["attention_backend"], "flash_attn")
        self.assertEqual(flags["optimization"], "serving")
        self.assertFalse(flags["enable_chunked_prefill"])
        self.assertEqual(flags["max_prefill_tokens"], 4096)
        self.assertEqual(self.engine._jetspec_scheduler.max_num_seqs, 4)

    def test_helper_reconfiguration_stays_flash_and_restores_originals(self):
        original_mode = qualification.shared.set_mode
        original_chunk = qualification.chunks.configure
        with qualification.flash_helpers(self.engine, self.args):
            qualification.shared.set_mode(self.engine, "jetspec", self.args.draft, 2)
            qualification.chunks.configure(self.engine, self.args, 4, 64)
            self.assertTrue(all(flags["attention_backend"] == "flash_attn" for _, flags in self.engine.calls))
            self.assertTrue(self.engine.calls[-1][1]["enable_chunked_prefill"])
            self.assertEqual(self.engine.calls[-1][1]["max_prefill_tokens"], 64)
        self.assertIs(qualification.shared.set_mode, original_mode)
        self.assertIs(qualification.chunks.configure, original_chunk)

    def test_nonidle_configuration_is_rejected(self):
        self.engine.idle = False
        with self.assertRaisesRegex(AssertionError, "idle"):
            qualification.configure(self.engine, self.args)
        self.assertFalse(self.engine.calls)

    def test_sdpa_control_is_explicit_not_fallback(self):
        qualification.configure(self.engine, self.args, backend="sdpa")
        self.assertEqual(self.engine.calls[-1][1]["attention_backend"], "sdpa")
        with self.assertRaisesRegex(AssertionError, "unknown"):
            qualification.configure(self.engine, self.args, backend="automatic")

    def test_serving_prefill_cannot_enable_rejected_flash_backend(self):
        self.engine._jetspec_scheduler.runtime._prefill_attention_backend = "flash_attn"
        with self.assertRaisesRegex(AssertionError, "unqualified"):
            qualification.configure(self.engine, self.args)

    def test_independent_causal_gqa_oracle_has_exact_first_value(self):
        generator = torch.Generator().manual_seed(3)
        q = torch.randn(4, 4, 8, generator=generator)
        k, v = [torch.randn(4, 2, 8, generator=generator) for _ in range(2)]
        expected = qualification.causal_fp64(q, k, v, 8 ** -.5)
        self.assertTrue(torch.equal(expected[0], v[0].double().repeat_interleave(2, dim=0)))
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[2:] += 7
        changed_v[2:] -= 3
        changed = qualification.causal_fp64(q, changed_k, changed_v, 8 ** -.5)
        self.assertTrue(torch.equal(expected[:2], changed[:2]))
        self.assertFalse(torch.equal(expected[2:], changed[2:]))

    def test_lifecycle_requires_actual_eos_not_just_cap_completion(self):
        rows = [dict(request_id=name, status=status, token_ids=tokens, max_tokens=64)
                for name, status, tokens in [("eos", "finished", [2, 99]),
                    ("running-cancel", "cancelled", [3]), ("queued-cancel", "cancelled", []),
                    ("one-token", "finished", [4]), ("late", "finished", [5]), ("refill", "finished", [6])]]
        result = dict(requests=rows, dynamic_live_arrival_seen=True)
        self.assertTrue(all(qualification.lifecycle_semantics(result, {99}).values()))
        rows[0]["token_ids"] = [2, 1]
        with self.assertRaisesRegex(AssertionError, "EOS"):
            qualification.lifecycle_semantics(result, {99})


if __name__ == "__main__":
    unittest.main()
