"""Independent CPU numerical oracles; no model weights or CUDA required."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.jetspec_numeric_oracles import (
    argmax_flip_witness, fp64_attention, numerical_metrics, parent_chain,
)


class NumericOracleCPU(unittest.TestCase):
    def test_parent_chain_independent_of_off_branch_and_mask(self):
        self.assertEqual(parent_chain([-1, 0, 0, 1, 2, 3], 5), [0, 1, 3, 5])
        self.assertEqual(parent_chain(torch.tensor([-1, 0, 0]), 0), [0])
        self.assertEqual(parent_chain([-1, 0, 999, 1], 3), [0, 1, 3])

    def test_invalid_parent_chains_fail_loudly(self):
        for parents, selected in (([-1, 2, 1], 1), ([-1, 4], 1),
                                  ([-1, -1], 1), ([0, 0], 1), ([-1], 1)):
            with self.subTest(parents=parents, selected=selected):
                with self.assertRaises(ValueError):
                    parent_chain(parents, selected)

    def test_fp64_attention_matches_hand_computed_gqa(self):
        q = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
        k = torch.tensor([[[1., 0.], [0., 0.]], [[0., 1.], [1., 1.]]])
        v = torch.tensor([[[2., 4.], [8., 10.]], [[6., 8.], [12., 14.]]])
        actual = fp64_attention(q, k, v, 1.0, 2)
        p = torch.softmax(torch.tensor([1., 0.], dtype=torch.float64), dim=0)
        group1_p = torch.softmax(torch.tensor([0., 2.], dtype=torch.float64), dim=0)
        expected = torch.stack((p[0] * v[0, 0] + p[1] * v[1, 0],
                                p[1] * v[0, 0] + p[0] * v[1, 0],
                                group1_p[0] * v[0, 1] + group1_p[1] * v[1, 1],
                                (v[0, 1] + v[1, 1]) / 2)).double()
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)
        self.assertEqual(actual.dtype, torch.float64)

    def test_fp64_attention_rejects_bad_geometry_and_nonfinite(self):
        q = torch.ones(2, 4)
        k = torch.ones(3, 1, 4)
        for args in ((q, k, k, 1., 1), (q, k[:0], k[:0], 1., 2),
                     (q, k, k, float("inf"), 2),
                     (q, k * float("nan"), k, 1., 2)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    fp64_attention(*args)

    def test_metrics_preserve_fp64_sub_float32_difference(self):
        reference = torch.tensor([1., 2.], dtype=torch.float64)
        actual = reference + 1e-9
        result = numerical_metrics(actual, reference)
        self.assertGreater(result["max_abs_error"], 0.)
        self.assertAlmostEqual(result["max_abs_error"], 1e-9, places=15)
        self.assertTrue(result["within_fixed_bound"])
        self.assertTrue(result["fixed_bound_applicable"])

    def test_fp32_fixed_bound_is_not_relaxed_for_bf16_rounding(self):
        reference = torch.tensor([1.003, -1.003], dtype=torch.float64)
        bf16 = reference.bfloat16()
        result = numerical_metrics(bf16, reference)
        self.assertFalse(result["within_fixed_bound"])
        self.assertFalse(result["fixed_bound_applicable"])
        self.assertEqual(result["bf16_quantization"]["nearest_bf16_unequal_elements"], 0)
        self.assertEqual(result["bf16_quantization"]["max_distance_from_nearest_bf16_in_ulps"], 0.)

    def test_bf16_one_ulp_midpoint_crossing_is_reported_separately(self):
        reference = torch.tensor([1.00390625 - 1e-9], dtype=torch.float64)
        actual = torch.tensor([1.0078125], dtype=torch.bfloat16)
        result = numerical_metrics(actual, reference)
        self.assertEqual(result["bf16_quantization"]["nearest_bf16_unequal_elements"], 1)
        self.assertEqual(result["bf16_quantization"]["max_distance_from_nearest_bf16_in_ulps"], 1.)

    def test_argmax_flip_witness_handles_near_tie_and_robust_margin(self):
        reference = torch.tensor([[1.000000001, 1., 0.], [3., 1., 0.]], dtype=torch.float64)
        actual = torch.tensor([[1., 1.000000001, 0.], [3.000000001, 1., 0.]], dtype=torch.float64)
        result = argmax_flip_witness(actual, reference)
        self.assertEqual(result["argmax_flips"], 1)
        self.assertEqual(result["large_margin_violations"], 0)
        self.assertTrue(result["rows"][0]["flip_within_error_envelope"])
        self.assertGreater(result["rows"][1]["reference_top2_margin"], result["rows"][1]["two_delta"])

    def test_argmax_tie_order_uses_first_index_not_topk(self):
        result = argmax_flip_witness(torch.tensor([34., 34.]), torch.tensor([33.75, 34.]))
        self.assertEqual(result["rows"][0]["actual_argmax"], 0)
        self.assertEqual(result["rows"][0]["reference_argmax"], 1)
        self.assertTrue(result["rows"][0]["flip_within_error_envelope"])

    def test_metric_and_logit_nonfinite_inputs_rejected(self):
        with self.assertRaises(ValueError):
            numerical_metrics(torch.tensor([float("nan")]), torch.ones(1))
        with self.assertRaises(ValueError):
            argmax_flip_witness(torch.tensor([float("inf"), 0.]), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
