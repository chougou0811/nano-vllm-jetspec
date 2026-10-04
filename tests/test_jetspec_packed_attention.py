"""CPU metadata gates and opt-in CUDA FP32 packed-attention qualification.

RUN_JETSPEC_PACKED_GPU_TESTS=1 python -m unittest discover -s tests \
    -p test_jetspec_packed_attention.py -v

Default test discovery runs only metadata checks, so it never competes with a
model benchmark for the GPU. Synthetic CUDA tests do not load model weights.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata


def ancestor_mask(nodes: int, device="cpu") -> torch.Tensor:
    result = torch.zeros((nodes, nodes), dtype=torch.bool)
    for node in range(nodes):
        current = node
        while current >= 0:
            result[node, current] = True
            current = (current - 1) // 2
    return result.to(device)


def metadata_fixture(device="cpu", counts=(63, 31, 47)):
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    slots = [torch.arange(6 * 256 + starts[i], 6 * 256 + starts[i + 1],
                          dtype=torch.int64, device=device) for i in range(len(counts))]
    prefixes = (65, 0, 257)[:len(counts)]
    pages = ((2,), (), (4, 0))[:len(counts)]
    masks = [ancestor_mask(count, device) for count in counts]
    metadata = PackedTreeMetadata.build(prefixes, pages, slots, masks, 256)
    return metadata, slots, masks


class PackedMetadataCPU(unittest.TestCase):
    def test_exact_ragged_offsets_and_local_masks(self):
        metadata, _, _ = metadata_fixture()
        self.assertEqual(metadata.query_offsets, (0, 63, 94, 141))
        self.assertEqual(metadata.request_slice(1), slice(63, 94))
        self.assertEqual(metadata.prefix_lengths, (65, 0, 257))
        self.assertEqual(metadata.node_counts_host, (63, 31, 47))
        self.assertEqual(metadata.mask_elements, 63 ** 2 + 31 ** 2 + 47 ** 2)
        self.assertLess(metadata.mask_elements, metadata.total_queries ** 2)
        self.assertEqual(metadata.block_tables.tolist(), [[2, -1], [-1, -1], [4, 0]])
        self.assertEqual(metadata.query_to_request.tolist(), [0] * 63 + [1] * 31 + [2] * 47)
        self.assertEqual(metadata.query_local_row.tolist(), list(range(63)) + list(range(31)) + list(range(47)))
        self.assertEqual(metadata.report()["numerical_contract"], "fp32_online_softmax_tile64")
        with self.assertRaises(IndexError):
            metadata.request_slice(-1)
        with self.assertRaises(IndexError):
            metadata.request_slice(3)

    def test_rejects_overlapping_scratch_and_committed_pages(self):
        _, slots, masks = metadata_fixture()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            PackedTreeMetadata.build([65, 0], [[2], []], [slots[0], slots[0][:31]], masks[:2], 256)
        with self.assertRaisesRegex(ValueError, "committed"):
            PackedTreeMetadata.build([65], [[6]], [slots[0]], masks[:1], 256)

    def test_rejects_noncanonical_tables_and_malformed_ancestors(self):
        _, slots, masks = metadata_fixture()
        with self.assertRaisesRegex(ValueError, "canonical"):
            PackedTreeMetadata.build([257], [[2]], slots[:1], masks[:1], 256)
        with self.assertRaisesRegex(ValueError, "canonical"):
            PackedTreeMetadata.build([257], [[2, 2]], slots[:1], masks[:1], 256)
        future = masks[0].clone()
        future[0, 1] = True
        with self.assertRaisesRegex(ValueError, "causal"):
            PackedTreeMetadata.build([65], [[2]], slots[:1], [future], 256)
        missing_self = masks[0].clone()
        missing_self[1, 1] = False
        with self.assertRaisesRegex(ValueError, "self"):
            PackedTreeMetadata.build([65], [[2]], slots[:1], [missing_self], 256)

    def test_pool_bounds_and_geometry(self):
        metadata, _, _ = metadata_fixture()
        metadata.validate_pool_geometry(torch.empty(2, 3, 8, 256, 2, 4))
        with self.assertRaisesRegex(ValueError, "geometry"):
            metadata.validate_pool_geometry(torch.empty(2, 3, 8, 16, 2, 4))
        with self.assertRaisesRegex(ValueError, "outside"):
            metadata.validate_pool_geometry(torch.empty(2, 3, 6, 256, 2, 4))

    def test_legacy_entry_cannot_silently_switch_multi_request_to_bf16(self):
        from nanovllm.speculative.jetspec.paged_backend import paged_tree_attention

        with self.assertRaisesRegex(ValueError, "PackedTreeMetadata"):
            paged_tree_attention(torch.empty(2, 2, 4), torch.empty(1, 256, 1, 4),
                                 torch.empty(1, 256, 1, 4), torch.empty(2, 1, dtype=torch.int32),
                                 torch.tensor([0, 1, 2], dtype=torch.int32),
                                 torch.tensor([1, 1], dtype=torch.int32), None, 0.5, 2, 256)


GPU_ENABLED = os.getenv("RUN_JETSPEC_PACKED_GPU_TESTS") == "1"


@unittest.skipUnless(GPU_ENABLED, "set RUN_JETSPEC_PACKED_GPU_TESTS=1 for CUDA qualification")
class PackedAttentionCUDA(unittest.TestCase):
    def fixture(self, counts=(63, 31, 47), dtype=torch.bfloat16):
        metadata, slots, masks = metadata_fixture("cuda", counts)
        generator = torch.Generator(device="cuda").manual_seed(317)
        head_dim, kv_heads, groups = 128, 2, 4
        q = torch.randn((metadata.total_queries, kv_heads * groups, head_dim),
                        generator=generator, device="cuda", dtype=dtype)
        k = torch.full((8, 256, kv_heads, head_dim), float("nan"), device="cuda", dtype=dtype)
        v = torch.full_like(k, float("nan"))
        for request, prefix in enumerate(metadata.prefix_lengths):
            positions = torch.arange(prefix, dtype=torch.int64, device="cuda")
            table = torch.tensor(metadata.block_tables_host[request], dtype=torch.int64, device="cuda")
            prefix_slots = table[positions // 256] * 256 + positions % 256
            live = torch.cat((prefix_slots, slots[request]))
            for pool in (k, v):
                pool[live // 256, live % 256] = torch.randn(
                    (live.numel(), kv_heads, head_dim), generator=generator, device="cuda", dtype=dtype)
        return q, k, v, metadata, slots, masks, groups

    def independent(self, q, k, v, metadata, slots, masks, groups):
        from nanovllm.speculative.jetspec.paged_backend import paged_tree_attention

        outputs = []
        for index, prefix in enumerate(metadata.prefix_lengths):
            local_q = q[metadata.request_slice(index)]
            positions = torch.arange(prefix, device="cuda", dtype=torch.int64)
            table = torch.tensor(metadata.block_tables_host[index], device="cuda", dtype=torch.int64)
            prefix_slots = table[positions // 256] * 256 + positions % 256
            logical = torch.cat((prefix_slots, slots[index]))
            total = prefix + local_q.shape[0]
            bias = torch.where(masks[index], 0.0, float("-inf")).float()
            # Use the exact legacy c1's 16-token view of the same 256-page pool.
            outputs.append(paged_tree_attention(
                local_q, k.view(-1, 16, k.shape[2], k.shape[3]),
                v.view(-1, 16, v.shape[2], v.shape[3]),
                torch.zeros((1, (total + 15) // 16), device="cuda", dtype=torch.int32),
                torch.tensor([0, local_q.shape[0]], device="cuda", dtype=torch.int32),
                torch.tensor([total], device="cuda", dtype=torch.int32),
                bias, q.shape[-1] ** -0.5, groups, 16,
                logical.reshape(1, -1), torch.tensor([0], device="cuda", dtype=torch.int32),
                torch.tensor([total], device="cuda", dtype=torch.int32),
            ))
        return torch.cat(outputs)

    def assert_bytes_equal(self, actual, expected):
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                    expected.contiguous().view(torch.uint8)))

    def test_c1_c2_c3_c4_c8_use_identical_fp32_attention_arithmetic(self):
        from nanovllm.speculative.jetspec.paged_backend import packed_tree_attention

        for counts in ((63,), (63, 31), (63, 31, 47)):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(counts=counts, dtype=dtype):
                    q, k, v, metadata, slots, masks, groups = self.fixture(counts, dtype)
                    packed = packed_tree_attention(q, k, v, metadata, 128 ** -0.5, groups)
                    expected = self.independent(q, k, v, metadata, slots, masks, groups)
                    self.assertTrue(torch.isfinite(packed).all().item(), "NaN padding was read")
                    self.assert_bytes_equal(packed, expected)
        # c4/c8 share one page where possible and use uneven node counts/prefixes.
        for count in (4, 8):
            q, k, v, base, _, _, groups = self.fixture()
            node_counts = [63, 31, 47, 1, 15, 2, 7, 3][:count]
            slots = []
            cursor = 6 * 256
            for n in node_counts:
                slots.append(torch.arange(cursor, cursor + n, device="cuda", dtype=torch.int64))
                cursor += n
            prefixes = [65, 0, 257, 1, 0, 64, 256, 3][:count]
            # Prefix sharing here only exercises read-only addressing: all pages
            # are initialized separately from the one packed scratch workspace.
            pages = [[2], [], [4, 0], [2], [], [2], [4], [0]][:count]
            masks = [ancestor_mask(n, "cuda") for n in node_counts]
            metadata = PackedTreeMetadata.build(prefixes, pages, slots, masks, 256)
            torch.manual_seed(123)
            q = torch.randn((sum(node_counts), 8, 128), device="cuda", dtype=torch.bfloat16)
            k = torch.randn_like(k)
            v = torch.randn_like(v)
            packed = packed_tree_attention(q, k, v, metadata, 128 ** -0.5, groups)
            expected = self.independent(q, k, v, metadata, slots, masks, groups)
            self.assert_bytes_equal(packed, expected)

    def test_other_request_prefix_tree_and_queries_cannot_affect_neighbors(self):
        from nanovllm.speculative.jetspec.paged_backend import packed_tree_attention

        q, k, v, metadata, slots, _, groups = self.fixture()
        original = packed_tree_attention(q, k, v, metadata, 128 ** -0.5, groups)
        # Perturb request A's prefix, nodes and queries, preserving packed shape.
        changed_q, changed_k, changed_v = q.clone(), k.clone(), v.clone()
        changed_q[metadata.request_slice(0)] *= 31
        prefix_positions = torch.arange(65, device="cuda")
        prefix_slots = 2 * 256 + prefix_positions
        changed_slots = torch.cat((prefix_slots, slots[0]))
        changed_k[changed_slots // 256, changed_slots % 256] *= -17
        changed_v[changed_slots // 256, changed_slots % 256] += 53
        result = packed_tree_attention(changed_q, changed_k, changed_v, metadata, 128 ** -0.5, groups)
        for neighbor in (1, 2):
            self.assert_bytes_equal(result[metadata.request_slice(neighbor)],
                                    original[metadata.request_slice(neighbor)])
        self.assertFalse(torch.equal(result[metadata.request_slice(0)], original[metadata.request_slice(0)]))

    def test_off_branch_finite_kv_changes_do_not_affect_ancestor_path(self):
        from nanovllm.speculative.jetspec.paged_backend import packed_tree_attention

        q, k, v, metadata, slots, masks, groups = self.fixture()
        original = packed_tree_attention(q, k, v, metadata, 128 ** -0.5, groups)
        selected_node = 55
        ancestors = masks[0][selected_node]
        changed_k, changed_v = k.clone(), v.clone()
        off_path = slots[0][~ancestors]
        changed_k[off_path // 256, off_path % 256] *= 97
        changed_v[off_path // 256, off_path % 256] += 113
        result = packed_tree_attention(q, changed_k, changed_v, metadata, 128 ** -0.5, groups)
        self.assert_bytes_equal(result[:63][ancestors], original[:63][ancestors])


if __name__ == "__main__":
    unittest.main()
