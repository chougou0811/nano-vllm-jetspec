"""HF-order RoPE/scatter CPU validation and opt-in CUDA byte-exact gates.

RUN_JETSPEC_TREE_FUSION_GPU_TESTS=1 python -m unittest discover -s tests \
    -p test_jetspec_tree_fusion.py -v

Normal discovery does not initialize CUDA or load model weights.
"""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.speculative.jetspec.tree_fusion import rope_scatter, rope_scatter_prevalidated


def eager_rope(q, k, positions, cache):
    selected = cache[positions]
    if selected.ndim == 2:
        selected = selected.unsqueeze(1)
    cos, sin = selected.chunk(2, -1)
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)

    def rotate(x):
        x1, x2 = x.chunk(2, -1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)

    return rotate(q), rotate(k)


def bits_equal(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape and
            torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def fixture(device="cpu", dtype=torch.bfloat16, *, dim=128, strided=False, cache_rank=3):
    generator = torch.Generator(device=device).manual_seed(740319)
    rows, q_heads, kv_heads = 7, 8, 2

    def values(heads):
        if strided:
            backing = torch.randn(rows * 2, heads * 2, dim * 2, generator=generator,
                                  device=device, dtype=dtype)
            return backing[::2, ::2, ::2]
        return torch.randn(rows, heads, dim, generator=generator, device=device, dtype=dtype)

    q, k, v = values(q_heads), values(kv_heads), values(kv_heads)
    positions = torch.tensor([0, 1, 127, 255, 513, 1023, 2047], device=device)
    slots = torch.tensor([257, 1, 767, 256, 0, 513, 511], device=device)
    if strided:
        position_backing = torch.zeros(rows * 2, dtype=torch.int64, device=device)
        slot_backing = torch.zeros_like(position_backing)
        position_backing[::2], slot_backing[::2] = positions, slots
        positions, slots = position_backing[::2], slot_backing[::2]
    angles = torch.arange(2048, device=device, dtype=torch.float32)[:, None] * (
        torch.arange(dim // 2, device=device, dtype=torch.float32)[None, :] + 1) * .00317
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    if strided:
        backing = torch.empty(cache.shape[0] * 2, dim * 2, dtype=cache.dtype, device=device)
        backing[::2, ::2] = cache
        cache = backing[::2, ::2]
    if cache_rank == 3:
        cache = cache.unsqueeze(1)
    # The two slices share one native [K/V,pages,page_size,heads,dim] allocation,
    # exactly like production. Their address ranges must be disjoint.
    pools = torch.randn(2, 4, 256, kv_heads, dim, generator=generator,
                        device=device, dtype=dtype)
    return q, k, v, positions, cache, pools[0], pools[1], slots


class TreeFusionCPUValidation(unittest.TestCase):
    def test_cpu_invocation_cannot_launch_cuda(self):
        with self.assertRaisesRegex(ValueError, "CUDA tensors"):
            rope_scatter(*fixture())

    def test_odd_or_too_large_head_dimension_is_rejected(self):
        for dim in (1, 127, 258):
            args = list(fixture())
            args[:3] = [torch.empty(7, heads, dim, dtype=torch.bfloat16) for heads in (8, 2, 2)]
            with self.subTest(dim=dim), self.assertRaisesRegex(ValueError, "even head geometry"):
                rope_scatter(*args)

    def test_qkv_dtype_and_grouping_mismatch_is_rejected(self):
        args = list(fixture())
        args[1] = args[1].float()
        with self.assertRaisesRegex(ValueError, "matching BF16 or FP32"):
            rope_scatter(*args)
        args = list(fixture())
        args[0] = args[0][:, :3]
        with self.assertRaisesRegex(ValueError, "GQA"):
            rope_scatter(*args)

    def test_bad_integer_vector_or_cache_layout_is_rejected(self):
        for index in (3, 7):
            args = list(fixture())
            args[index] = args[index].float()
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "integer"):
                rope_scatter(*args)
        args = list(fixture())
        args[4] = args[4].expand(-1, 2, -1)
        with self.assertRaisesRegex(ValueError, "cos/sin cache"):
            rope_scatter(*args)

    def test_launch_only_boundary_does_not_download_or_allocate(self):
        args = fixture(strided=True)
        out = torch.empty_like(args[0])
        with patch("nanovllm.speculative.jetspec.tree_fusion._rope_scatter_hf_order") as kernel:
            received = rope_scatter_prevalidated(*args, out)
        self.assertIs(received, out)
        self.assertEqual(kernel.__getitem__.call_args.args[0], (7, 8))
        self.assertFalse(kernel.__getitem__.return_value.call_args.kwargs["enable_fp_fusion"])

    def test_bf16_multiply_rounding_matters_before_add(self):
        # A focused witness guarding against replacing four BF16 products by a
        # fused FP32 expression with one output cast.
        q = torch.tensor([[[1.0078125, 1.015625]]], dtype=torch.bfloat16)
        cache = torch.tensor([[.70703125, .70703125]], dtype=torch.float32)
        eager, _ = eager_rope(q, q, torch.tensor([0]), cache)
        x1, x2 = q.float().chunk(2, -1)
        cos, sin = cache.to(q.dtype).float().unsqueeze(1).chunk(2, -1)
        single_cast = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).to(q.dtype)
        self.assertFalse(bits_equal(eager, single_cast))


@unittest.skipUnless(os.environ.get("RUN_JETSPEC_TREE_FUSION_GPU_TESTS") == "1",
                     "enable explicit CUDA RoPE/scatter qualification")
class TreeFusionGPU(unittest.TestCase):
    def assert_equivalent(self, args, *, provided_out=False):
        q, k, v, positions, cache, k_pool, v_pool, slots = args
        expected_q, expected_k = eager_rope(q, k, positions, cache)
        expected_k_pool, expected_v_pool = k_pool.clone(), v_pool.clone()
        pages, offsets = slots // k_pool.shape[1], slots % k_pool.shape[1]
        expected_k_pool[pages, offsets] = expected_k
        expected_v_pool[pages, offsets] = v
        out = torch.empty_like(q) if provided_out else None
        actual = rope_scatter(*args, out=out)
        self.assertTrue(bits_equal(actual, expected_q), "Q RoPE must match eager bytes")
        self.assertTrue(bits_equal(k_pool, expected_k_pool), "including every untouched K slot")
        self.assertTrue(bits_equal(v_pool, expected_v_pool), "including every untouched V slot")
        if provided_out:
            self.assertIs(actual, out)
        return actual

    def test_native_bf16_fp32_contiguous_and_strided_byte_exact(self):
        for dtype in (torch.bfloat16, torch.float32):
            for dim in (32, 96, 128, 256):
                for strided in (False, True):
                    for rank in (2, 3):
                        with self.subTest(dtype=dtype, dim=dim, strided=strided, rank=rank):
                            self.assert_equivalent(fixture("cuda", dtype, dim=dim,
                                                           strided=strided, cache_rank=rank),
                                                   provided_out=strided)

    def test_dynamic_positions_slots_replay_and_finite_other_request_changes(self):
        args = list(fixture("cuda", strided=True))
        for _ in range(3):
            self.assert_equivalent(args)
            args[3].copy_((args[3] + 17) % args[4].shape[0])
            args[7].copy_((args[7] + 43) % (args[5].shape[0] * args[5].shape[1]))
        baseline = self.assert_equivalent(args).clone()
        args[0][3:].mul_(17)
        args[1][3:].add_(8)
        args[2][3:].sub_(5)
        modified = self.assert_equivalent(args)
        self.assertTrue(bits_equal(modified[:3], baseline[:3]), "other request cannot change Q")

    def test_rounding_witness_and_signed_zero(self):
        args = list(fixture("cuda", dim=32))
        args[0].fill_(1.0078125)
        args[0][..., 16:] = 1.015625
        args[1].copy_(args[0][:, :2])
        args[4][..., :16] = .70703125
        args[4][..., 16:] = .70703125
        self.assert_equivalent(args)
        args[0].zero_()
        args[0][..., 16:] = -0.0
        args[1].copy_(args[0][:, :2])
        self.assert_equivalent(args)

    def test_bf16_cache_and_nan_unallocated_pages_are_untouched(self):
        args = list(fixture("cuda", strided=True))
        args[4] = args[4].to(torch.bfloat16)
        args[5].fill_(float("nan"))
        args[6].fill_(float("nan"))
        self.assert_equivalent(args)

    def test_invalid_addresses_duplicates_or_aliases_rejected_before_store(self):
        for name in ("position", "negative_slot", "too_large_slot", "duplicate"):
            args = list(fixture("cuda"))
            k_before, v_before = args[5].clone(), args[6].clone()
            if name == "position":
                args[3][0] = args[4].shape[0]
            elif name == "negative_slot":
                args[7][0] = -1
            elif name == "too_large_slot":
                args[7][0] = args[5].shape[0] * args[5].shape[1]
            else:
                args[7][0] = args[7][1]
            with self.subTest(name=name), self.assertRaises(ValueError):
                rope_scatter(*args)
            self.assertTrue(bits_equal(args[5], k_before))
            self.assertTrue(bits_equal(args[6], v_before))
        args = fixture("cuda")
        with self.assertRaisesRegex(ValueError, "output must not alias"):
            rope_scatter(*args, out=args[0])
        args = list(fixture("cuda"))
        args[6] = args[5]
        with self.assertRaisesRegex(ValueError, "pool views must not overlap"):
            rope_scatter(*args)
        args = fixture("cuda")
        overlap = torch.empty_like(args[0]).as_strided(args[0].shape, (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "overlapping write addresses"):
            rope_scatter(*args, out=overlap)


if __name__ == "__main__":
    unittest.main()
