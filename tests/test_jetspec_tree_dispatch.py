"""CPU-only production dispatch gates; no model or CUDA allocation."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from nanovllm.speculative.jetspec import paged_backend, tree_attention, tree_prefix


class Tensor:
    def __init__(self, shape, dtype=torch.bfloat16):
        self.shape, self.dtype = shape, dtype
        self.ndim, self.is_cuda, self.device = len(shape), True, torch.device("cuda:0")

    def stride(self, axis):
        axis %= self.ndim
        result = 1
        for value in self.shape[axis + 1:]:
            result *= value
        return result


class Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return call


def inputs(prefixes=(128, 1024)):
    count = len(prefixes)
    meta = SimpleNamespace(total_queries=63 * count, block_size=256,
        prefix_lengths=prefixes, num_requests=count, query_to_request=object(),
        query_local_row=object(), prefix_lens=object(), node_counts=object(),
        cu_seqlens_q=object(), block_tables=Tensor((count, 16)),
        tree_slots=Tensor((63 * count,)), qq_bias=object(), qq_bias_offsets=object())
    return (Tensor((63 * count, 32, 128)), Tensor((30, 256, 8, 128)),
            Tensor((30, 256, 8, 128)), meta, .125, 4)


class TreeDispatchTests(unittest.TestCase):
    def test_auto_uses_measured_prefix_path_without_changing_operands(self):
        args, marker = inputs(), object()
        with patch.object(paged_backend, "_tree_attention_device_capability", return_value=(12, 0)), \
                patch.object(tree_prefix, "packed_tree_attention_prefix_split_exact", return_value=marker) as call:
            self.assertIs(paged_backend.packed_tree_attention(*args), marker)
        call.assert_called_once_with(*args, output_dtype=None, num_warps=4)

    def test_c1_short_prefix_and_ragged_boundary_selection(self):
        with patch.object(paged_backend, "_tree_attention_device_capability", return_value=(12, 0)):
            for prefixes, expected in (((0,), False), ((128,), False), ((255,), False),
                    ((256,), True), ((2048,), True), ((0, 63), False), ((0, 64), True)):
                q, k, v, meta, _, groups = inputs(prefixes)
                with self.subTest(prefixes=prefixes):
                    self.assertEqual(paged_backend._use_prefix_tree_attention(q, k, v, meta, groups), expected)

    def test_unsupported_device_dtype_geometry_remain_reference(self):
        for cap in ((8, 0), (9, 0), (12, 1)):
            with patch.object(paged_backend, "_tree_attention_device_capability", return_value=cap):
                args = inputs()
                self.assertFalse(paged_backend._use_prefix_tree_attention(*args[:4], args[-1]))
        for dtype in (torch.float16, torch.float32):
            args = inputs()
            args[0].dtype = args[1].dtype = args[2].dtype = dtype
            self.assertFalse(paged_backend._use_prefix_tree_attention(*args[:4], args[-1]))
        args = inputs()
        args[3].block_size = 128
        self.assertFalse(paged_backend._use_prefix_tree_attention(*args[:4], args[-1]))
        args = inputs()
        args[0].shape = (126, 16, 128)
        self.assertFalse(paged_backend._use_prefix_tree_attention(*args[:4], args[-1]))

    def test_auto_fallback_launches_exact_frozen_kernel(self):
        args, kernel = inputs((128,)), Launch()
        marker = Tensor(args[0].shape)
        with patch.object(paged_backend, "_packed_paged_tree_fp32", kernel), \
                patch.object(paged_backend.torch, "empty_like", return_value=marker), \
                patch.object(paged_backend, "_tree_attention_device_capability") as hardware:
            self.assertIs(paged_backend.packed_tree_attention(*args), marker)
        hardware.assert_not_called()
        self.assertEqual(kernel.calls[0][0], (63, 32))
        self.assertEqual(kernel.calls[0][2]["TILE"], 64)

    def test_reference_override_does_not_consult_auto_predicate(self):
        args, marker = inputs(), object()
        with patch.object(tree_attention, "packed_tree_attention_reference", return_value=marker) as call, \
                patch.object(paged_backend, "_use_prefix_tree_attention") as predicate:
            self.assertIs(paged_backend.packed_tree_attention(*args, backend="reference",
                                                           output_dtype=torch.float32), marker)
        predicate.assert_not_called()
        call.assert_called_once_with(*args, output_dtype=torch.float32)

    def test_explicit_prefix_supports_pre_store_fp32_diagnostic(self):
        args, marker = inputs(), object()
        with patch.object(tree_prefix, "packed_tree_attention_prefix_split_exact", return_value=marker) as call:
            self.assertIs(paged_backend.packed_tree_attention(*args, backend="prefix",
                                                           output_dtype=torch.float32), marker)
        call.assert_called_once_with(*args, output_dtype=torch.float32, num_warps=4)

    def test_invalid_backend_output_dtype_fail_before_kernel(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            paged_backend.packed_tree_attention(*inputs(), backend="tensorcore")
        with self.assertRaisesRegex(ValueError, "output dtype"):
            paged_backend.packed_tree_attention(*inputs(), output_dtype=torch.float64)

    def test_device_capability_query_is_cached(self):
        paged_backend._tree_attention_device_capability.cache_clear()
        try:
            with patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)) as call:
                for _ in range(36):
                    self.assertEqual(paged_backend._tree_attention_device_capability(0), (12, 0))
                call.assert_called_once_with(0)
        finally:
            paged_backend._tree_attention_device_capability.cache_clear()


if __name__ == "__main__":
    unittest.main()
