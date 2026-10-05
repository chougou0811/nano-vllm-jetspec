"""CPU ownership, topology, numerical-contract and measurement gates."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import jetspec_tree_kernel_benchmark as bench


class TreeKernelBenchmarkCPU(unittest.TestCase):
    def test_matrix_is_predeclared_and_contains_ragged_tree_sizes(self):
        matrix = bench.cases()
        self.assertEqual(len(matrix), 10)
        self.assertEqual({s.name for s in matrix},
            {f"c{c}_p{p}" for c in (1, 4, 8) for p in (128, 1024, 2048)} | {"c8_mixed_prefix"})
        for shape in matrix:
            self.assertEqual(shape.nodes, tuple(bench.COUNTS[i % 3] for i in range(len(shape.prefixes))))
            self.assertEqual((shape.block_size, shape.kv_heads, shape.groups, shape.head_size), (256, 8, 4, 128))

    def test_random_page_and_slot_ownership_is_exact_and_reproducible(self):
        shape = bench.cases()[-1]
        a, b = bench.layout(shape, 613), bench.layout(shape, 614)
        self.assertEqual(a, bench.layout(shape, 613))
        self.assertNotEqual(a, b)
        prefixes = [page for table in a["prefix_pages"] for page in table]
        slots = sum(a["tree_slots"], ())
        self.assertEqual(len(prefixes), len(set(prefixes)))
        self.assertEqual(len(slots), sum(shape.nodes))
        self.assertEqual(len(slots), len(set(slots)))
        self.assertFalse(set(prefixes).intersection(slot // shape.block_size for slot in slots))
        self.assertGreater(len(set(slot // shape.block_size for slot in slots)), 1)
        self.assertTrue(any(y != x + 1 for x, y in zip(slots, slots[1:])))
        for request, prefix in enumerate(shape.prefixes):
            logical = bench.prefix_slots(shape, a, request)
            self.assertEqual(len(logical), prefix)
            self.assertEqual(len(set(logical)), prefix)
            self.assertTrue(all(slot // shape.block_size in a["prefix_pages"][request] for slot in logical))

    def test_cpu_metadata_matches_parent_chains_and_request_boundaries(self):
        shape = bench.cases()[-1]
        ownership = bench.layout(shape, 613)
        meta, slots = bench.metadata(shape, ownership, "cpu")
        self.assertEqual(meta.node_counts_host, shape.nodes)
        self.assertEqual(meta.prefix_lengths, shape.prefixes)
        self.assertEqual(meta.tree_slot_ids, sum(ownership["tree_slots"], ()))
        for request, nodes in enumerate(shape.nodes):
            sl = meta.request_slice(request)
            self.assertEqual(meta.query_to_request[sl].tolist(), [request] * nodes)
            self.assertEqual(meta.query_local_row[sl].tolist(), list(range(nodes)))
            start = int(meta.qq_bias_offsets[request])
            mask = meta.qq_bias[start:start + nodes * nodes].view(nodes, nodes).isfinite()
            for row in range(nodes):
                self.assertEqual(mask[row].nonzero().flatten().tolist(), list(bench.chain(ownership["parents"][request], row)))

    def test_zero_prefix_and_boundary_slots(self):
        shape = bench.Shape("edge", (0, 255, 256, 257), (1, 2, 3, 4))
        ownership = bench.layout(shape, 6)
        self.assertEqual([len(table) for table in ownership["prefix_pages"]], [0, 1, 1, 2])
        meta, _ = bench.metadata(shape, ownership, "cpu")
        self.assertEqual(meta.total_queries, 10)
        self.assertEqual(meta.block_tables[0].tolist(), [-1, -1])

    def test_bad_shapes_and_non_parent_trees_fail(self):
        for shape in (bench.Shape("bad", (), ()), bench.Shape("bad", (0,), (0,)),
                      bench.Shape("bad", (-1,), (1,)), replace(bench.cases()[0], block_size=0)):
            with self.assertRaises(ValueError):
                bench.layout(shape, 1)
        with self.assertRaises(ValueError):
            bench.binary_parents(0)
        for parents in ((0,), (-1, 2, 1), (-1, -1)):
            with self.assertRaises(ValueError):
                bench.chain(parents, len(parents) - 1)

    def test_fp32_gate_is_not_bf16_envelope(self):
        actual = torch.tensor([1., 2., 3.])
        within = bench.error_metrics(actual + 1e-5, actual.double())
        beyond = bench.error_metrics(actual + 1e-3, actual.double())
        self.assertTrue(within["passed"])
        self.assertFalse(beyond["passed"])
        self.assertEqual(within["relative_rms_bound"], 1e-4)
        self.assertEqual(bench.FP32_BOUND, 1e-4)
        self.assertFalse(bench.error_metrics(torch.tensor([float("nan")]), torch.ones(1))["passed"])

    def test_median_retains_outlier_and_rejects_fewer_than_three(self):
        raw = [{"cuda_event_span_ms_per_call": value, "synchronous_wall_ms_per_call": 2 * value}
               for value in (100, 2, 1)]
        result = bench.summarize(raw)
        self.assertEqual(result["median_cuda_event_span_ms_per_call"], 2)
        self.assertEqual(result["median_synchronous_wall_ms_per_call"], 4)
        self.assertEqual(len(raw), 3)
        with self.assertRaises(ValueError):
            bench.summarize(raw[:2])

    def test_oracle_reconstructs_from_parent_and_ignores_unowned_pages(self):
        shape = bench.Shape("oracle", (2, 0), (3, 2), block_size=4, kv_heads=1, groups=2, head_size=2)
        ownership = bench.layout(shape, 10)
        q = torch.ones((5, 2, 2))
        k = torch.ones((ownership["num_pages"], 4, 1, 2))
        v = torch.zeros_like(k)
        # All legal values are constant seven. Guard slots have huge finite poison.
        v.fill_(999)
        for request in range(2):
            for slot in bench.prefix_slots(shape, ownership, request) + ownership["tree_slots"][request]:
                v[slot // 4, slot % 4] = 7
        rows = bench.oracle_rows(q, k, v, shape, ownership)
        self.assertEqual([row for row, _ in rows], [0, 1, 2, 3, 4])
        for _, output in rows:
            torch.testing.assert_close(output, torch.full_like(output, 7))


if __name__ == "__main__":
    unittest.main()
