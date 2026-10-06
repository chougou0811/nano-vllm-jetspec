"""RMSNorm fusion metadata tests plus opt-in unchanged numerical gates.

RUN_JETSPEC_TREE_NORM_GPU_TESTS=1 python -m unittest discover -s tests \\
    -p test_jetspec_tree_norm.py -v

The GPU tests emit diagnostic differences and can fail: this is an explicit
prototype, not evidence that an untested full-network fusion is qualified.
"""
from pathlib import Path
import json
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nanovllm.speculative.jetspec import tree_norm


def eager_pre_store(x, weight, eps):
    hidden = x.float()
    variance = hidden.pow(2).mean(-1, keepdim=True)
    normalized = (hidden * torch.rsqrt(variance + eps)).to(x.dtype)
    return normalized.float() * weight.float()


def metrics(actual, expected, bound):
    a, b = actual.double(), expected.double()
    delta = a - b
    max_abs = float(delta.abs().max())
    scale = max(float(b.abs().max()), 1e-30)
    rms = float(delta.square().mean().sqrt())
    denominator = max(float(b.square().mean().sqrt()), 1e-30)
    return {"max_abs": max_abs, "scaled_max": max_abs / scale,
            "relative_rms": rms / denominator, "bound": bound,
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "passed": (bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                       and max_abs / scale <= bound and rms / denominator <= bound)}


class TreeNormMetadataCPU(unittest.TestCase):
    def setUp(self):
        self.x = torch.ones(2, 128, dtype=torch.bfloat16)
        self.weight = torch.ones(128, dtype=torch.bfloat16)

    def test_invalid_rank_and_width(self):
        for x in (torch.ones(1, 1, 1, 128, dtype=torch.bfloat16),
                  torch.ones(2, 0, dtype=torch.bfloat16),
                  torch.ones(2, 8193, dtype=torch.bfloat16)):
            with self.subTest(shape=x.shape), self.assertRaisesRegex(ValueError, "dimensions|width"):
                tree_norm.rms_norm(x, self.weight, 1e-6)

    def test_weight_geometry(self):
        for weight in (self.weight[:127], self.weight.view(1, 128)):
            with self.assertRaisesRegex(ValueError, "weight must be a vector"):
                tree_norm.rms_norm(self.x, weight, 1e-6)

    def test_dtype_validation(self):
        for x, weight in ((self.x.half(), self.weight.half()),
                          (self.x, self.weight.float()),
                          (self.x.long(), self.weight.long())):
            with self.assertRaisesRegex(ValueError, "BF16 or FP32"):
                tree_norm.rms_norm(x, weight, 1e-6)

    def test_eps_must_be_positive_and_finite(self):
        for eps in (True, False, 0, -1e-6, float("nan"), float("inf"), "1e-6"):
            with self.subTest(eps=eps), self.assertRaisesRegex(ValueError, "eps"):
                tree_norm.rms_norm(self.x, self.weight, eps)

    def test_kernel_options_and_output_dtype(self):
        for warps in (True, 0, 1, 3, 16):
            with self.assertRaisesRegex(ValueError, "num_warps"):
                tree_norm.rms_norm(self.x, self.weight, 1e-6, num_warps=warps)
        with self.assertRaisesRegex(ValueError, "output dtype"):
            tree_norm.rms_norm(self.x, self.weight, 1e-6, output_dtype=torch.float16)

    def test_valid_cpu_input_cannot_silently_fall_back(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            tree_norm.rms_norm(self.x, self.weight, 1e-6)

    def test_reference_signature_forwards_weight_and_eps(self):
        norm = SimpleNamespace(weight=self.weight, eps=1e-6)
        sentinel = object()
        with patch.object(tree_norm, "rms_norm", return_value=sentinel) as launch:
            self.assertIs(tree_norm.reference_rms_norm(self.x, norm), sentinel)
        launch.assert_called_once_with(self.x, self.weight, 1e-6)

    def test_intermediate_bf16_rounding_is_observable(self):
        x = torch.tensor([[1.5, 0.2, -0.1, 2.0]], dtype=torch.bfloat16)
        weight = torch.tensor([1.01, 0.99, 1.2, 0.8], dtype=torch.bfloat16)
        hidden = x.float()
        normalized_fp32 = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
        correct = eager_pre_store(x, weight, 1e-6)
        incorrectly_merged = normalized_fp32 * weight.float()
        self.assertFalse(torch.equal(correct, incorrectly_merged))
        self.assertTrue(torch.equal(correct.to(torch.bfloat16),
            weight * normalized_fp32.to(torch.bfloat16)))

    def test_fixed_diagnostic_bounds_reject_large_error(self):
        self.assertTrue(metrics(torch.ones(8), torch.ones(8), 1e-4)["passed"])
        self.assertFalse(metrics(torch.ones(8) + 2e-4, torch.ones(8), 1e-4)["passed"])
        self.assertFalse(metrics(torch.ones(8) + .02, torch.ones(8), 2 ** -6)["passed"])
        self.assertFalse(metrics(torch.full((8,), float("nan")), torch.ones(8), 2 ** -6)["passed"])


@unittest.skipUnless(os.environ.get("RUN_JETSPEC_TREE_NORM_GPU_TESTS") == "1",
                     "explicit RMSNorm GPU qualification is disabled")
class TreeNormGPU(unittest.TestCase):
    def test_native_operands_strided_layouts_and_fixed_eager_gates(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        torch.manual_seed(53)
        records = []
        for dtype in (torch.bfloat16, torch.float32):
            for shape in ((128,), (63, 128), (31, 8, 128), (47, 32, 128),
                          (63, 4096), (7, 128), (7, 129)):
                for strided in (False, True):
                    storage_shape = (*shape[:-1], shape[-1] * 2) if strided else shape
                    storage = torch.randn(storage_shape, dtype=dtype, device="cuda")
                    x = storage[..., ::2] if strided else storage
                    if x.ndim == 3 and strided:
                        # Transpose outer rows/heads as well as striding D.
                        x = x.transpose(0, 1)
                    weight_storage = torch.randn(shape[-1] * (2 if strided else 1),
                                                 dtype=dtype, device="cuda")
                    weight = weight_storage[::2] if strided else weight_storage
                    reference = eager_pre_store(x, weight, 1e-6)
                    diagnostic = tree_norm.rms_norm(x, weight, 1e-6, output_dtype=torch.float32)
                    actual = tree_norm.rms_norm(x, weight, 1e-6)
                    fp32 = metrics(diagnostic, reference, 1e-4)
                    native = metrics(actual, reference.to(dtype), 2 ** -6)
                    row = {"dtype": str(dtype), "shape": list(x.shape),
                           "strides": list(x.stride()), "weight_stride": weight.stride(0),
                           "pre_store_fp32": fp32, "native": native,
                           "native_unequal_elements": int((actual != reference.to(dtype)).sum()),
                           "repeat_exact": torch.equal(actual, tree_norm.rms_norm(x, weight, 1e-6))}
                    records.append(row)
        # Print all witnesses before assertions; do not hide or relax a failed
        # synthetic diagnostic. Full trained-model gates remain separate.
        print("RMSNORM_PROTOTYPE_DIAGNOSTICS " + json.dumps(records))
        self.assertTrue(all(r["pre_store_fp32"]["passed"] for r in records), records)
        self.assertTrue(all(r["native"]["passed"] for r in records), records)
        self.assertTrue(all(r["repeat_exact"] for r in records), records)

    def test_request_row_isolation_and_empty_rows(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        x = torch.randn(63, 8, 128, dtype=torch.bfloat16, device="cuda")
        weight = torch.ones(128, dtype=x.dtype, device=x.device)
        before = tree_norm.rms_norm(x, weight, 1e-6)
        x[31:] += 17
        after = tree_norm.rms_norm(x, weight, 1e-6)
        self.assertTrue(torch.equal(before[:31], after[:31]))
        empty = tree_norm.rms_norm(x[:0], weight, 1e-6)
        self.assertEqual(tuple(empty.shape), (0, 8, 128))
        self.assertEqual(empty.dtype, x.dtype)


if __name__ == "__main__":
    unittest.main()
