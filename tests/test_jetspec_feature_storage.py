"""CPU gates for append-friendly features and authoritative host acceptance."""
import unittest
from unittest.mock import patch

import torch

from nanovllm.speculative.jetspec import state as state_module
from nanovllm.speculative.jetspec.state import BatchTreeTransaction
from tests.test_jetspec_batch_state import make_requests, commit_inputs, fill_tree, read_slots


class JetSpecFeatureStorageTest(unittest.TestCase):
    def setUp(self):
        self.states, self.arena, self.manager = make_requests(prefix_lengths=(3, 3))

    def tearDown(self):
        if self.arena._batch_transaction is not None:
            self.arena._batch_transaction.abort()
        for state in self.states:
            state.clear()
        self.arena.clear()
        self.assertFalse(self.manager.used_block_ids)

    def admit(self, paths, *, counts=None, maxima=None):
        counts = [5] * len(self.states) if counts is None else counts
        maxima = [len(path) for path in paths] if maxima is None else maxima
        tx = BatchTreeTransaction.admit(self.states, counts, maxima, self.arena)
        for index, nodes in enumerate(tx.node_slots):
            fill_tree(self.arena, nodes, index)
        tensor_paths = [torch.tensor(path, dtype=torch.long) for path in paths]
        hidden, tokens = commit_inputs(self.states, counts, tensor_paths)
        return tx, tensor_paths, hidden, tokens

    def originals(self):
        return [(state.target_hidden, state.target_hidden.clone(), state._feature_storage,
                 state.feature_storage_snapshot(), state.committed, state.logical_slots)
                for state in self.states]

    def assert_unpublished(self, before):
        for state, (view, values, storage, metrics, tokens, slots) in zip(self.states, before):
            self.assertIs(state.target_hidden, view)
            self.assertTrue(torch.equal(state.target_hidden, values))
            self.assertIs(state._feature_storage, storage)
            self.assertEqual(state.feature_storage_snapshot(), metrics)
            self.assertIs(state.committed, tokens)
            self.assertIs(state.logical_slots, slots)

    def test_legacy_default_keeps_concat_values_and_new_copy_accounting(self):
        previous = [state.target_hidden.clone() for state in self.states]
        tx, paths, hidden, tokens = self.admit([[0, 2], [0]])
        report = tx.commit(hidden, paths, tokens)
        self.assertEqual(report["feature_history_copy_bytes"], 2 * 3 * 6 * 4)
        self.assertEqual(report["feature_append_copy_bytes"], 3 * 6 * 4)
        for state, old, features, path in zip(self.states, previous, hidden, paths):
            self.assertFalse(state.feature_storage_snapshot()["feature_storage_enabled"])
            self.assertTrue(torch.equal(state.target_hidden, torch.cat((old, features.index_select(1, path)), 1)))
            state.assert_round_invariant()

    def test_enable_default_reuses_old_tensor_and_explicit_reserve_copies_once(self):
        first, second = self.states
        original = first.target_hidden
        report = first.enable_feature_storage()
        self.assertIs(first._feature_storage, original)
        self.assertEqual(first.target_hidden.data_ptr(), original.data_ptr())
        self.assertEqual(report["feature_history_copy_bytes"], 0)
        self.assertEqual(first.enable_feature_storage(), report)
        before = second.target_hidden.clone()
        report = second.enable_feature_storage(initial_capacity=16, max_capacity=16)
        self.assertEqual(second.feature_capacity, 16)
        self.assertEqual(report["feature_history_copy_bytes"], 3 * 6 * 4)
        self.assertEqual(report["feature_reserved_bytes"], 16 * 6 * 4)
        self.assertEqual(report["feature_growths"], 1)
        self.assertTrue(torch.equal(second.target_hidden, before))
        second.assert_round_invariant()

    def test_buffered_append_never_concatenates_feature_history(self):
        for state in self.states:
            state.enable_feature_storage(initial_capacity=16)
        before = self.originals()
        tx, paths, hidden, tokens = self.admit([[0, 2], [0, 4]])
        original_cat = torch.cat

        def no_feature_cat(tensors, *args, **kwargs):
            tensors = tuple(tensors)
            if tensors and tensors[0].ndim == 3:
                raise AssertionError("feature history torch.cat is forbidden")
            return original_cat(tensors, *args, **kwargs)

        with patch.object(torch, "cat", side_effect=no_feature_cat):
            report = tx.commit(hidden, paths, tokens)
        self.assertEqual(report["feature_history_copy_bytes"], 0)
        for state, old, features, path in zip(self.states, before, hidden, paths):
            self.assertIs(state._feature_storage, old[2])
            self.assertTrue(torch.equal(old[0], old[1]))
            self.assertTrue(torch.equal(state.target_hidden[:, :3], old[1]))
            self.assertTrue(torch.equal(state.target_hidden[:, 3:], features.index_select(1, path)))
            self.assertEqual(state.target_hidden.data_ptr(), state._feature_storage.data_ptr())
            self.assertEqual(state.feature_capacity, 16)
            state.assert_round_invariant()

    def test_bf16_appended_features_preserve_exact_bits_and_byte_accounting(self):
        for state in self.states:
            state.target_hidden = state.target_hidden.to(torch.bfloat16)
            state.enable_feature_storage(initial_capacity=16)
        before = [state.target_hidden.clone() for state in self.states]
        tx, paths, hidden, tokens = self.admit([[0, 2], [0, 4]])
        hidden = [features.to(torch.bfloat16) for features in hidden]
        report = tx.commit(hidden, paths, tokens)
        self.assertEqual(report["feature_append_copy_bytes"], 4 * 6 * 2)
        self.assertEqual(report["feature_history_copy_bytes"], 0)
        for state, old, features, path in zip(self.states, before, hidden, paths):
            expected = torch.cat((old, features.index_select(1, path)), 1)
            self.assertEqual(state.target_hidden.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(state.target_hidden.contiguous().view(torch.uint8),
                                        expected.contiguous().view(torch.uint8)))
            state.assert_round_invariant()

    def test_empty_prefix_feature_view_has_valid_backing_and_can_append(self):
        for state in self.states:
            state.clear()
        self.arena.clear()
        self.states, self.arena, self.manager = make_requests(prefix_lengths=(0, 0))
        for state in self.states:
            state.enable_feature_storage(initial_capacity=4)
            self.assertEqual(tuple(state.target_hidden.shape), (1, 0, 6))
            state.assert_round_invariant()
        tx, paths, hidden, tokens = self.admit([[0], [0]])
        tx.commit(hidden, paths, tokens)
        for state, features in zip(self.states, hidden):
            self.assertTrue(torch.equal(state.target_hidden, features[:, :1]))
            state.assert_round_invariant()

    def test_geometric_growth_preserves_prior_views_and_is_bounded(self):
        for state in self.states:
            state.enable_feature_storage(max_capacity=8)
        old = self.originals()
        tx, paths, hidden, tokens = self.admit([[0, 2], [0]])
        report = tx.commit(hidden, paths, tokens)
        self.assertEqual(report["feature_history_copy_bytes"], 2 * 3 * 6 * 4)
        for state, previous in zip(self.states, old):
            self.assertEqual(state.feature_capacity, 6)
            self.assertIsNot(state._feature_storage, previous[2])
            self.assertTrue(torch.equal(previous[0], previous[1]))
            self.assertEqual(state.feature_storage_snapshot()["feature_growths"], 1)
        tx, paths, hidden, tokens = self.admit([[0, 1], [0]])
        report = tx.commit(hidden, paths, tokens)
        self.assertEqual(self.states[0].feature_capacity, 8)
        self.assertEqual(self.states[1].feature_capacity, 6)
        self.assertEqual(report["feature_history_copy_bytes"], 5 * 6 * 4)

    def test_long_append_replay_migrates_history_only_at_geometric_growth(self):
        for state in self.states:
            state.enable_feature_storage()
        originals = [state.target_hidden.clone() for state in self.states]
        expected = [tensor.clone() for tensor in originals]
        old_views = []
        for round_id in range(1000):
            tx, paths, hidden, tokens = self.admit([[0], [0]], counts=[1, 1])
            hidden = [features + round_id for features in hidden]
            old_views.append((self.states[0].target_hidden, self.states[0].target_hidden.clone()))
            report = tx.commit(hidden, paths, tokens)
            self.assertEqual(report["feature_append_copy_bytes"], 2 * 6 * 4)
            expected = [torch.cat((old, new), 1) for old, new in zip(expected, hidden)]
        for state, features in zip(self.states, expected):
            self.assertTrue(torch.equal(state.target_hidden, features))
            state.assert_round_invariant()
            metrics = state.feature_storage_snapshot()
            self.assertLess(metrics["feature_history_copy_bytes"], 2 * metrics["feature_live_tokens"] * 6 * 4)
            self.assertLessEqual(metrics["feature_capacity_tokens"], 2 * metrics["feature_live_tokens"])
            self.assertLessEqual(metrics["feature_growths"], 10)
            self.assertEqual(metrics["feature_append_copy_bytes"], 1000 * 6 * 4)
        self.assertTrue(all(torch.equal(view, saved) for view, saved in old_views))

    def test_invalid_last_path_leaves_earlier_prepared_tail_unpublished(self):
        for state in self.states:
            state.enable_feature_storage(initial_capacity=16)
        before = self.originals()
        tx, paths, hidden, tokens = self.admit([[0, 2], [0, 7]])
        with patch.object(state_module, "copy_accepted_kv") as copy:
            with self.assertRaisesRegex(ValueError, "bounds"):
                tx.commit(hidden, paths, tokens)
        copy.assert_not_called()
        self.assert_unpublished(before)
        tx.abort()

    def test_feature_growth_allocation_error_publishes_nothing_after_other_tail_write(self):
        self.states[0].enable_feature_storage(initial_capacity=16)
        self.states[1].enable_feature_storage()
        before = self.originals()
        tx, paths, hidden, tokens = self.admit([[0, 2], [0, 4]])
        original_empty = torch.empty

        def fail_feature_allocation(size, *args, **kwargs):
            if tuple(size) == (1, 6, 6):
                raise RuntimeError("injected feature growth allocation failure")
            return original_empty(size, *args, **kwargs)

        with patch.object(torch, "empty", side_effect=fail_feature_allocation), patch.object(
            state_module, "copy_accepted_kv"
        ) as copy:
            with self.assertRaisesRegex(RuntimeError, "feature growth allocation"):
                tx.commit(hidden, paths, tokens)
        copy.assert_not_called()
        self.assert_unpublished(before)
        tx.abort()

    def test_partial_kv_copy_error_leaves_features_unpublished_and_retry_overwrites_tail(self):
        for state in self.states:
            state.enable_feature_storage(initial_capacity=16)
        before = self.originals()
        tx, paths, hidden, tokens = self.admit([[0, 2], [0, 4]])
        prefixes = [read_slots(self.arena.kv_pool, state.logical_slots) for state in self.states]
        original_copy = state_module.copy_accepted_kv

        def fail_copy(*args):
            original_copy(*args)
            raise RuntimeError("injected after KV copy")

        with patch.object(state_module, "copy_accepted_kv", side_effect=fail_copy):
            with self.assertRaisesRegex(RuntimeError, "after KV copy"):
                tx.commit(hidden, paths, tokens)
        self.assert_unpublished(before)
        self.assertTrue(all(torch.equal(read_slots(self.arena.kv_pool, state.logical_slots), old)
                            for state, old in zip(self.states, prefixes)))
        tx.abort()
        tx, paths, hidden, tokens = self.admit([[0], [0]])
        hidden = [features + 123 for features in hidden]
        tx.commit(hidden, paths, tokens)
        for state, feature in zip(self.states, hidden):
            self.assertTrue(torch.equal(state.target_hidden[:, -1:], feature[:, :1]))

    def test_interrupted_publication_restores_storage_views_and_counters(self):
        for state in self.states:
            state.enable_feature_storage()
        before = self.originals()
        tx, paths, hidden, tokens = self.admit([[0, 2], [0, 4]])
        with patch.object(self.states[1], "_publish_features", side_effect=KeyboardInterrupt("feature publication")):
            with self.assertRaisesRegex(KeyboardInterrupt, "feature publication"):
                tx.commit(hidden, paths, tokens)
        self.assertFalse(tx.committed)
        self.assert_unpublished(before)
        tx.abort()
        for state in self.states:
            state.assert_round_invariant()

    def test_unused_page_release_error_rolls_back_feature_growth_publication(self):
        self.states[0].clear()
        self.states[1].clear()
        self.arena.clear()
        self.states, self.arena, self.manager = make_requests(prefix_lengths=(253, 253))
        for state in self.states:
            state.enable_feature_storage()
        before = self.originals()
        tx, paths, hidden, tokens = self.admit([[0], [0]], maxima=[4, 4])
        with patch.object(self.manager, "_deallocate_block", side_effect=KeyboardInterrupt("unused release")):
            with self.assertRaisesRegex(KeyboardInterrupt, "unused release"):
                tx.commit(hidden, paths, tokens)
        self.assertFalse(tx.committed)
        self.assert_unpublished(before)
        tx.abort()

    def test_interrupt_after_completed_release_preserves_new_feature_views_and_counters(self):
        for state in self.states:
            state.enable_feature_storage()
        old = [state.target_hidden.clone() for state in self.states]
        tx, paths, hidden, tokens = self.admit([[0, 2], [0, 4]])
        original_release = self.manager.release_provisional

        def fail_after_release(blocks):
            original_release(blocks)
            raise KeyboardInterrupt("after completed release")

        with patch.object(self.manager, "release_provisional", side_effect=fail_after_release):
            with self.assertRaisesRegex(KeyboardInterrupt, "completed release"):
                tx.commit(hidden, paths, tokens)
        self.assertTrue(tx.committed)
        self.assertFalse(tx.active)
        for state, previous, features, path in zip(self.states, old, hidden, paths):
            self.assertTrue(torch.equal(state.target_hidden, torch.cat((previous, features.index_select(1, path)), 1)))
            self.assertEqual(state.feature_storage_snapshot()["feature_growths"], 1)
            self.assertEqual(state.feature_storage_snapshot()["feature_append_copy_bytes"], 2 * 6 * 4)
            state.assert_round_invariant()

    def test_feature_maximum_capacity_error_precedes_copy_and_publication(self):
        for state in self.states:
            state.enable_feature_storage(initial_capacity=4, max_capacity=4)
        before = self.originals()
        tx, paths, hidden, tokens = self.admit([[0, 2], [0]])
        with patch.object(state_module, "copy_accepted_kv") as copy:
            with self.assertRaisesRegex(RuntimeError, "capacity limit"):
                tx.commit(hidden, paths, tokens)
        copy.assert_not_called()
        self.assert_unpublished(before)
        tx.abort()

    def test_host_paths_are_authoritative_and_do_not_call_tensor_tolist(self):
        for state in self.states:
            state.enable_feature_storage(initial_capacity=16)
        tx, paths, hidden, tokens = self.admit([[0, 2], [0, 4]])
        expected = [read_slots(self.arena.kv_pool, nodes[path]) for nodes, path in zip(tx.node_slots, paths)]
        stale_paths = [torch.tensor([0, 1]), torch.tensor([0, 2])]
        with patch.object(torch.Tensor, "tolist", side_effect=AssertionError("no path download allowed")):
            tx.commit(hidden, stale_paths, tokens, accepted_paths_host=[[0, 2], [0, 4]])
        for state, raw, features, actual_path in zip(self.states, expected, hidden, paths):
            self.assertTrue(torch.equal(read_slots(self.arena.kv_pool, state.logical_slots[-2:]), raw))
            self.assertTrue(torch.equal(state.target_hidden[:, -2:], features.index_select(1, actual_path)))
            state.assert_round_invariant()

    def test_invalid_host_paths_never_copy_or_publish(self):
        invalid = ([], [1], [0, 0], [0, -1], [0, 5], [0, True], [0, 1.0], [0, "1"])
        for bad in invalid:
            with self.subTest(path=bad):
                tx, paths, hidden, tokens = self.admit([[0], [0]])
                paths[0] = torch.zeros(len(bad), dtype=torch.long)
                tokens[0] = torch.zeros(1, self.states[0].cache_len + len(bad) + 1, dtype=torch.long)
                with patch.object(state_module, "copy_accepted_kv") as copy:
                    with self.assertRaises(ValueError):
                        tx.commit(hidden, paths, tokens, accepted_paths_host=[bad, [0]])
                copy.assert_not_called()
                tx.abort()

    def test_host_path_count_and_supplied_tensor_shape_must_align(self):
        tx, paths, hidden, tokens = self.admit([[0], [0]])
        with self.assertRaisesRegex(ValueError, "align"):
            tx.commit(hidden, paths, tokens, accepted_paths_host=[[0]])
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            tx.commit(hidden, paths, tokens, accepted_paths_host=[[0, 2], [0]])
        tx.abort()

    def test_lightweight_flag_skips_only_canonical_device_validation_and_force_restores_it(self):
        state = self.states[0]
        state.validate_device = False
        with patch.object(torch, "equal", side_effect=AssertionError("device equality was called")):
            state.assert_round_invariant()
        original = state.logical_slots.clone()
        state.logical_slots[0] = state.logical_slots[1]
        state.assert_round_invariant()
        with self.assertRaisesRegex(RuntimeError, "not canonical"):
            state.assert_round_invariant(validate_device=True)
        state.logical_slots = original
        state.validate_device = True
        state.assert_round_invariant()

    def test_lightweight_checks_still_enforce_lengths_ownership_and_feature_views(self):
        state = self.states[0]
        state.validate_device = False
        original = state.committed
        state.committed = state.committed[:, :-1]
        with self.assertRaisesRegex(RuntimeError, "invariant failed"):
            state.assert_round_invariant()
        state.committed = original
        block = self.manager.blocks[state.owned_blocks[0]]
        block.ref_count = 0
        try:
            with self.assertRaisesRegex(RuntimeError, "lost paged KV ownership"):
                state.assert_round_invariant()
        finally:
            block.ref_count = 1
        state.enable_feature_storage(initial_capacity=8)
        view = state.target_hidden
        state.target_hidden = view.clone()
        with self.assertRaisesRegex(RuntimeError, "backing capacity"):
            state.assert_round_invariant()
        state.target_hidden = view
        state.assert_round_invariant()

    def test_enable_validation_and_clear_release_feature_capacity(self):
        state = self.states[0]
        for kwargs in ({"initial_capacity": 2}, {"initial_capacity": True},
                       {"initial_capacity": 8, "max_capacity": 7}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                state.enable_feature_storage(**kwargs)
        old_view = state.target_hidden
        old_values = old_view.clone()
        state.enable_feature_storage(initial_capacity=16)
        state.clear()
        self.assertEqual(state.feature_capacity, 0)
        self.assertEqual(state.feature_storage_snapshot()["feature_copy_bytes"], 0)
        self.assertIsNone(state._feature_storage)
        self.assertTrue(torch.equal(old_view, old_values))
        with self.assertRaisesRegex(RuntimeError, "cleared"):
            state.enable_feature_storage()

    def test_single_request_api_can_append_to_feature_capacity_without_semantic_change(self):
        state = self.states[0]
        state.enable_feature_storage(initial_capacity=16)
        original = state.target_hidden.clone()
        nodes, _ = state.reserve_tree(5, max_path_length=2)
        hidden = torch.arange(30, dtype=state.target_hidden.dtype).reshape(1, 5, 6)
        path = torch.tensor([0, 2])
        tokens = torch.cat((state.committed, torch.full((1, 2), 17, dtype=torch.long)), 1)
        report = state.commit_tree_path(nodes, hidden, path, committed_tokens=tokens)
        self.assertEqual(report["feature_history_copy_bytes"], 0)
        self.assertEqual(report["feature_append_copy_bytes"], 2 * 6 * 4)
        self.assertTrue(torch.equal(state.target_hidden, torch.cat((original, hidden.index_select(1, path)), 1)))
        state.assert_round_invariant()


if __name__ == "__main__":
    unittest.main()
