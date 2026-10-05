from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from nanovllm.speculative.jetspec import tree_dot


class FakeTensor:
    def __init__(self, shape, dtype=torch.bfloat16):
        self.shape, self.dtype = shape, dtype

    def stride(self, axis):
        result = 1
        for length in self.shape[axis + 1:]:
            result *= length
        return result


class Launch:
    def __init__(self):
        self.grid = self.args = self.kwargs = None

    def __getitem__(self, grid):
        self.grid = grid
        def launch(*args, **kwargs):
            self.args, self.kwargs = args, kwargs
        return launch


def inputs(dtype=torch.bfloat16, groups=4):
    meta = SimpleNamespace(block_size=256, total_queries=94,
        node_counts_host=(63, 31), num_requests=2,
        prefix_lens=object(), node_counts=object(), cu_seqlens_q=object(),
        block_tables=FakeTensor((2, 10)), tree_slots=object(),
        qq_bias=object(), qq_bias_offsets=object())
    return (FakeTensor((94, 8 * groups, 128), dtype),
            FakeTensor((20, 256, 8, 128), dtype),
            FakeTensor((20, 256, 8, 128), dtype), meta, .125, groups)


class TreeDotTests(unittest.TestCase):
    def launch(self, dtype=torch.bfloat16, query_tile=4, groups=4):
        args = inputs(dtype, groups)
        output, kernel = FakeTensor(args[0].shape, torch.float32), Launch()
        with patch.object(tree_dot, "_validate", return_value=output) as validate, \
                patch.object(tree_dot, "_packed_tree_dot_fp32", kernel):
            result = tree_dot.packed_tree_attention_dot(*args, output_dtype=torch.float32,
                                                       query_tile=query_tile)
        validate.assert_called_once_with(args[0], args[1], args[2], args[3], groups, torch.float32)
        self.assertIs(result, output)
        return kernel

    def test_default_launch_groups_four_queries_per_gqa_head(self):
        kernel = self.launch()
        self.assertEqual(kernel.grid, (16, 8, 2))
        self.assertEqual(kernel.kwargs["ROWS"], 16)
        self.assertEqual(kernel.kwargs["QUERY_TILE"], 4)
        self.assertEqual(kernel.kwargs["GROUPS"], 4)
        self.assertEqual(kernel.kwargs["TILE"], 64)
        self.assertFalse(kernel.kwargs["Q_FP32"])

    def test_fp32_qk_selects_explicit_tf32x3_precision_branch(self):
        self.assertTrue(self.launch(torch.float32).kwargs["Q_FP32"])
        self.assertFalse(self.launch(torch.float16).kwargs["Q_FP32"])

    def test_small_query_tiles_pad_dot_rows_without_extra_visible_queries(self):
        for tile in (1, 2, 4, 8):
            for groups in (1, 2, 4, 8):
                kernel = self.launch(query_tile=tile, groups=groups)
                rows = kernel.kwargs["ROWS"]
                self.assertGreaterEqual(rows, 16)
                valid = [row for row in range(rows) if row < tile * groups]
                self.assertEqual(len(valid), tile * groups)
                self.assertEqual({row // groups for row in valid}, set(range(tile)))

    def test_non_power_of_two_gqa_explicit_reference_fallback(self):
        args = inputs(groups=3)
        expected = object()
        with patch.object(tree_dot, "packed_tree_attention_reference", return_value=expected) as reference, \
                patch.object(tree_dot, "_validate") as validate:
            result = tree_dot.packed_tree_attention_dot(*args, output_dtype=torch.float32)
        self.assertIs(result, expected)
        reference.assert_called_once_with(*args, output_dtype=torch.float32)
        validate.assert_not_called()

    def test_invalid_experiment_parameters_rejected(self):
        for tile in (True, 0, 3, 16, "4"):
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                tree_dot.packed_tree_attention_dot(*inputs(), query_tile=tile)
        for warps in (True, 0, 1, 2, 3, 32, "4"):
            with self.subTest(warps=warps), self.assertRaises(ValueError):
                tree_dot.packed_tree_attention_dot(*inputs(), num_warps=warps)


if __name__ == "__main__":
    unittest.main()
