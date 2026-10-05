import ast
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from nanovllm.speculative.jetspec import tree_prefix


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape

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


def metadata(block_size=256):
    tensor = FakeTensor((2, 10))
    return SimpleNamespace(block_size=block_size, total_queries=94,
        query_to_request=object(), query_local_row=object(), prefix_lens=object(),
        node_counts=object(), cu_seqlens_q=object(), block_tables=tensor,
        tree_slots=object(), qq_bias=object(), qq_bias_offsets=object())


class TreePrefixTests(unittest.TestCase):
    def inputs(self, block_size=256):
        return (FakeTensor((94, 32, 128)), FakeTensor((20, block_size, 8, 128)),
                FakeTensor((20, block_size, 8, 128)), metadata(block_size), .125, 4)

    def test_complete_prefix_tiles_do_not_cross_pages(self):
        for page_size in (64, 128, 256, 512):
            for prefix in (0, 1, 63, 64, 65, 255, 256, 257, 1024, 2049):
                full_tiles = prefix // 64
                for tile in range(full_tiles):
                    start = tile * 64
                    offsets = [start % page_size + k for k in range(64)]
                    self.assertTrue(all(0 <= off < page_size for off in offsets))
                    self.assertEqual(start // page_size, (start + 63) // page_size)
                    self.assertLess(start + 63, prefix)

    def test_split_preserves_every_original_tile_once(self):
        for prefix in (0, 1, 63, 64, 65, 255, 256, 257, 2048):
            for nodes in (1, 31, 47, 63, 65):
                complete = prefix // 64
                total = (prefix + nodes + 63) // 64
                split = list(range(complete)) + list(range(complete, total))
                self.assertEqual(split, list(range(total)))
                # The mixed tail starts on the OLD 64-key boundary, not at P.
                self.assertEqual(complete * 64, prefix - prefix % 64)

    def test_launcher_uses_scalar_grid_and_fixed_arithmetic_tile(self):
        args = self.inputs()
        output, kernel = FakeTensor((94, 32, 128)), Launch()
        with patch.object(tree_prefix, "_validate", return_value=output) as validate, \
                patch.object(tree_prefix, "_packed_tree_prefix_fp32", kernel):
            result = tree_prefix.packed_tree_attention_prefix(*args, output_dtype=torch.float32, num_warps=8)
        self.assertIs(result, output)
        validate.assert_called_once_with(args[0], args[1], args[2], args[3], 4, torch.float32)
        self.assertEqual(kernel.grid, (94, 32))
        self.assertEqual(kernel.kwargs["TILE"], 64)
        self.assertEqual(kernel.kwargs["BLOCK_SIZE"], 256)
        self.assertEqual(kernel.kwargs["GROUPS"], 4)
        self.assertEqual(kernel.kwargs["num_warps"], 8)

    def test_unsupported_page_geometry_explicitly_returns_reference(self):
        for page_size in (16, 32, 96):
            args = self.inputs(page_size)
            expected = object()
            with patch.object(tree_prefix, "packed_tree_attention_reference", return_value=expected) as reference, \
                    patch.object(tree_prefix, "_validate") as validate:
                result = tree_prefix.packed_tree_attention_prefix(*args, output_dtype=torch.float32)
            self.assertIs(result, expected)
            reference.assert_called_once_with(*args, output_dtype=torch.float32)
            validate.assert_not_called()

    def test_invalid_num_warps_rejected_before_launch(self):
        for value in (True, False, 0, 1, 2, 3, 32, "4", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tree_prefix.packed_tree_attention_prefix(*self.inputs(), num_warps=value)

    def test_validation_failure_does_not_launch(self):
        kernel = Mock()
        with patch.object(tree_prefix, "_validate", side_effect=ValueError("invalid Q/K/V")), \
                patch.object(tree_prefix, "_packed_tree_prefix_fp32", kernel):
            with self.assertRaisesRegex(ValueError, "invalid Q/K/V"):
                tree_prefix.packed_tree_attention_prefix(*self.inputs())
        kernel.assert_not_called()

    def test_single_loop_variant_uses_frozen_grid_tile_and_warps(self):
        args = self.inputs()
        output, kernel = FakeTensor((94, 32, 128)), Launch()
        with patch.object(tree_prefix, "_validate", return_value=output) as validate, \
                patch.object(tree_prefix, "_packed_tree_prefix_single_loop_fp32", kernel):
            result = tree_prefix.packed_tree_attention_prefix_single_loop(*args, output_dtype=torch.float32)
        self.assertIs(result, output)
        validate.assert_called_once_with(args[0], args[1], args[2], args[3], 4, torch.float32)
        self.assertEqual(kernel.grid, (94, 32))
        self.assertEqual(kernel.kwargs["TILE"], 64)
        self.assertEqual(kernel.kwargs["block_size"], 256)
        self.assertEqual(kernel.kwargs["num_queries_per_kv"], 4)
        self.assertEqual(kernel.kwargs["num_warps"], 4)

    def test_single_loop_arithmetic_body_is_frozen_reference_ast(self):
        from nanovllm.speculative.jetspec.paged_backend import _packed_paged_tree_fp32
        def source_loop(kernel):
            node = ast.parse(inspect.getsource(kernel.fn)).body[0]
            loops = [part for part in ast.walk(node) if isinstance(part, ast.For)]
            self.assertEqual(len(loops), 1)
            return loops[0]
        def arithmetic(loop):
            start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                         and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "k")
            return ast.dump(ast.Module(body=loop.body[start:], type_ignores=[]), include_attributes=False)
        reference = source_loop(_packed_paged_tree_fp32)
        candidate = source_loop(tree_prefix._packed_tree_prefix_single_loop_fp32)
        self.assertEqual(ast.dump(candidate.iter, include_attributes=False),
                         ast.dump(reference.iter, include_attributes=False))
        self.assertEqual(arithmetic(candidate), arithmetic(reference))

    def test_single_loop_geometry_explicit_fallback(self):
        args = self.inputs(32)
        expected = object()
        with patch.object(tree_prefix, "packed_tree_attention_reference", return_value=expected) as reference:
            result = tree_prefix.packed_tree_attention_prefix_single_loop(*args, output_dtype=torch.float32)
        self.assertIs(result, expected)
        reference.assert_called_once_with(*args, output_dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
