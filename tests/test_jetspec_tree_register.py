from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from nanovllm.speculative.jetspec import paged_backend, tree_attention


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
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
        return launch


def inputs():
    meta = SimpleNamespace(block_size=256, total_queries=94,
        query_to_request=object(), query_local_row=object(), prefix_lens=object(),
        node_counts=object(), cu_seqlens_q=object(), block_tables=FakeTensor((2, 10)),
        tree_slots=object(), qq_bias=object(), qq_bias_offsets=object())
    return (FakeTensor((94, 32, 128)), FakeTensor((20, 256, 8, 128)),
            FakeTensor((20, 256, 8, 128)), meta, .125, 4)


class TreeRegisterTests(unittest.TestCase):
    def test_reference_and_register_variant_launch_identical_frozen_kernel(self):
        args = inputs()
        out = FakeTensor(args[0].shape)
        kernel = Launch()
        with patch.object(tree_attention, "_validate", return_value=out), \
                patch.object(paged_backend, "_packed_paged_tree_fp32", kernel):
            reference = tree_attention.packed_tree_attention_reference(*args, output_dtype=torch.float32)
            tuned = tree_attention.packed_tree_attention_register_tuned(*args, output_dtype=torch.float32)
        self.assertIs(reference, out)
        self.assertIs(tuned, out)
        self.assertEqual(len(kernel.calls), 2)
        ref_grid, ref_args, ref_options = kernel.calls[0]
        new_grid, new_args, new_options = kernel.calls[1]
        self.assertEqual(new_grid, ref_grid)
        self.assertEqual(new_args, ref_args)
        self.assertEqual(new_options["num_warps"], 4)
        self.assertEqual(new_options["maxnreg"], 96)
        self.assertEqual({key: value for key, value in new_options.items()
                          if key not in ("num_warps", "maxnreg")}, ref_options)
        self.assertEqual(ref_options["TILE"], 64)

    def test_allowed_caps_are_only_extra_compiler_option(self):
        for cap in (64, 80, 96, 112, 128):
            with self.subTest(cap=cap):
                args, kernel = inputs(), Launch()
                with patch.object(tree_attention, "_validate", return_value=FakeTensor(args[0].shape)), \
                        patch.object(paged_backend, "_packed_paged_tree_fp32", kernel):
                    tree_attention.packed_tree_attention_register_tuned(*args, maxnreg=cap)
                self.assertEqual(kernel.calls[0][2]["maxnreg"], cap)
                self.assertEqual(kernel.calls[0][2]["num_warps"], 4)

    def test_cap_validation_precedes_any_allocation_launch(self):
        for cap in (True, False, None, 0, 63, 65, 79, 97, 129, "96"):
            with self.subTest(cap=cap), patch.object(tree_attention, "_validate") as validate:
                with self.assertRaisesRegex(ValueError, "maxnreg"):
                    tree_attention.packed_tree_attention_register_tuned(*inputs(), maxnreg=cap)
                validate.assert_not_called()

    def test_standard_validation_and_fp32_output_are_reused(self):
        args, kernel = inputs(), Launch()
        out = FakeTensor(args[0].shape)
        with patch.object(tree_attention, "_validate", return_value=out) as validate, \
                patch.object(paged_backend, "_packed_paged_tree_fp32", kernel):
            tree_attention.packed_tree_attention_register_tuned(*args, output_dtype=torch.float32)
        validate.assert_called_once_with(args[0], args[1], args[2], args[3], args[5], torch.float32)


if __name__ == "__main__":
    unittest.main()
