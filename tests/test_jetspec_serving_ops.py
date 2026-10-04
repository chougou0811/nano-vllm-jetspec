"""Semantic gates for round-wide transfers and lightweight serving decisions."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from jetspec.tree import get_algorithm, gpu_tree_accept
from nanovllm.speculative.jetspec.serving_ops import build_trees, accept_batch
from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata
from nanovllm.speculative.jetspec.state import BatchTreeTransaction
from tests.test_jetspec_batch_runtime import FakeEvent
from tests.test_jetspec_continuous import make_serving_runtime


class ServingOpsTest(unittest.TestCase):
    def trees(self):
        generator = torch.Generator().manual_seed(17)
        proposals = [torch.randn(1, 15, 80, generator=generator) for _ in range(4)]
        proposals[0] = None
        budgets = [1, 31, 47, 63]
        roots = [3, 5, 7, 9]
        trees = build_trees(roots, proposals, budgets, 15, 7, torch.device("cpu"))
        return trees, proposals, budgets, roots

    def test_batched_topk_heap_matches_official_tree_policy(self):
        trees, proposals, budgets, roots = self.trees()
        for tree, logits, budget, root in zip(trees, proposals, budgets, roots):
            if logits is None:
                self.assertEqual(tree.host_token_ids, [root])
                continue
            expected = get_algorithm("accum_logp").build(root, logits, 16, 7, budget, torch.device("cpu"))
            for key in ("token_ids", "parent_indices", "depth", "ancestor"):
                self.assertTrue(torch.equal(getattr(tree, key), getattr(expected, key)), key)
            self.assertEqual(tree.child_maps, expected.child_maps)

    def test_one_download_acceptance_matches_official_with_duplicate_siblings(self):
        trees, _, _, _ = self.trees()
        duplicate = SimpleNamespace(token_ids=torch.tensor([3, 5, 5, 7, 9]),
            parent_indices=torch.tensor([-1, 0, 0, 1, 2]), depth=torch.tensor([0, 1, 1, 2, 2]), num_nodes=5)
        trees.append(duplicate)
        offsets = [0]
        for tree in trees:
            offsets.append(offsets[-1] + tree.num_nodes)
        generator = torch.Generator().manual_seed(99)
        logits = torch.randn(offsets[-1], 80, generator=generator)
        logits[offsets[-2], 5] = 100
        logits[offsets[-2] + 2, 9] = 100
        results = accept_batch(logits, trees, offsets, 15)
        for i, (tree, result) in enumerate(zip(trees, results)):
            greedy = logits[offsets[i]:offsets[i + 1]].argmax(-1)
            path, accepted, correction = gpu_tree_accept(tree.token_ids, greedy,
                tree.parent_indices, tree.depth, max_depth=15)
            self.assertEqual(result["path"], path.tolist())
            self.assertEqual(result["accepted_length"], accepted)
            self.assertEqual(result["correction"], correction.item())
            self.assertEqual(result["outputs"], tree.token_ids[path[1:]].tolist() + [correction.item()])
        self.assertEqual(results[-1]["path"], [0, 2, 4])

    def test_host_metadata_matches_full_validation_every_field(self):
        trees, _, _, _ = self.trees()
        prefixes, tables = [5, 259, 27, 0], [[1], [3, 5], [8], []]
        arena, b = [11], 256
        ids = ["a", "b", "c", "d"]
        fast = PackedTreeMetadata.from_host_trees(prefixes, tables, trees, arena, b,
                                                   device="cpu", request_ids=ids)
        slots, offset = [], 0
        for tree in trees:
            slots.append(torch.arange(11 * b + offset, 11 * b + offset + tree.num_nodes))
            offset += tree.num_nodes
        full = PackedTreeMetadata.build(prefixes, tables, slots, [t.ancestor for t in trees], b, request_ids=ids)
        for key, value in vars(full).items():
            actual = getattr(fast, key)
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, actual), key)
            else:
                self.assertEqual(value, actual, key)

    def test_host_metadata_rejects_aliases_and_inconsistent_ancestry(self):
        trees, _, _, _ = self.trees()
        with self.assertRaisesRegex(ValueError, "overlap"):
            PackedTreeMetadata.from_host_trees([1] * 4, [[11]] * 4, trees, [11], 256,
                                               device="cpu", request_ids=list(range(4)))
        trees[1].host_depths[1] = 7
        with self.assertRaisesRegex(ValueError, "depth"):
            PackedTreeMetadata.from_host_trees([1] * 4, [[i] for i in range(4)], trees, [11], 256,
                                               device="cpu", request_ids=list(range(4)))


class LightweightRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.events = patch("torch.cuda.Event", FakeEvent)
        self.events.start()

    def tearDown(self):
        self.events.stop()

    def run_mode(self, lightweight, features):
        runtime = make_serving_runtime()
        runtime.tree_algorithm = get_algorithm("accum_logp")
        runtime.configure_optimizations(lightweight=lightweight, feature_storage=features)
        try:
            requests = [runtime.create_request([3, 4], max_new_tokens=12, tree_budget=cap,
                        request_id=str(i)) for i, cap in enumerate((31, 47))]
            records = []
            while not all(r.finished for r in requests):
                record = runtime.step([r for r in requests if not r.finished])
                records.append(record)
                for r in requests:
                    r.state.assert_round_invariant(validate_device=True)
            results = [runtime.finish(r)["token_ids"] for r in requests]
            return results, records
        finally:
            runtime.close()
            self.assertFalse(runtime.block_manager.used_block_ids)

    def test_lightweight_replay_matches_debug_and_keeps_required_stream_records(self):
        expected, old = self.run_mode(False, False)
        actual, new = self.run_mode(True, True)
        self.assertEqual(expected, actual)
        self.assertEqual(len(old), len(new))
        for record in new:
            self.assertNotIn("_verify_events", record)
            self.assertNotIn("_commit_events", record)
            for request in record["requests"]:
                self.assertNotIn("target_argmax_by_node", request)
                self.assertTrue(request["committed_path_indices"])
                self.assertTrue(request["output_block"])

    def test_policy_cannot_change_under_live_requests(self):
        runtime = make_serving_runtime()
        try:
            runtime.create_request([3], max_new_tokens=4, tree_budget=3)
            with self.assertRaisesRegex(RuntimeError, "live"):
                runtime.configure_optimizations(lightweight=True)
        finally:
            runtime.close()

    def test_lightweight_explicit_batch_wrapper_retains_requested_timing(self):
        runtime = make_serving_runtime()
        runtime.configure_optimizations(lightweight=True, feature_storage=True)
        try:
            with patch("torch.cuda.synchronize"):
                result = runtime.generate_batch([[3, 4], [5]], max_new_tokens=7,
                                                tree_budgets=3, ignore_eos=True)
            self.assertEqual(result["total_output_tokens"], 14)
            self.assertGreater(result["target_verification_latency_s"], 0)
            self.assertTrue(result["request_cleanup_passed"])
            for record in result["rounds"]:
                self.assertNotIn("_verify_events", record)
                self.assertIn("verify_latency_s", record)
                self.assertNotIn("target_argmax_by_node", record["requests"][0])
        finally:
            runtime.close()

    def test_lightweight_exception_boundaries_preserve_transaction_semantics(self):
        for postcommit in (False, True):
            with self.subTest(postcommit=postcommit):
                runtime = make_serving_runtime()
                runtime.configure_optimizations(lightweight=True, feature_storage=True)
                try:
                    requests = [runtime.create_request([3, 4], max_new_tokens=8,
                                tree_budget=3, request_id=i) for i in range(2)]
                    original = BatchTreeTransaction.commit if postcommit else runtime._verify_batch

                    def fail(*args, **kwargs):
                        original(*args, **kwargs)
                        raise RuntimeError("injected boundary failure")

                    owner, name = (BatchTreeTransaction, "commit") if postcommit else (runtime, "_verify_batch")
                    with patch.object(owner, name, fail), self.assertRaisesRegex(RuntimeError, "boundary"):
                        runtime.step(requests)
                    for request in requests:
                        request.state.assert_round_invariant(validate_device=True)
                        self.assertEqual(request.state.cache_len,
                                         request.prompt_length + len(request.output_ids) - 1)
                        self.assertEqual(bool(request.rounds), postcommit)
                        self.assertEqual(len(request.output_ids) > 1, postcommit)
                        self.assertFalse(request.state.pending_blocks)
                    self.assertIsNone(runtime._active_transaction)
                    self.assertFalse(runtime.arena.active)
                finally:
                    runtime.close()
                    self.assertFalse(runtime.block_manager.used_block_ids)


if __name__ == "__main__":
    unittest.main()
