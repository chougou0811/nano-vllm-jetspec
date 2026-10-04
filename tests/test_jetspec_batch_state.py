"""CPU gates for packed-batch KV admission, ownership and atomic commit."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.engine.block_manager import BlockManager
from nanovllm.speculative.jetspec import state as state_module
from nanovllm.speculative.jetspec.state import BatchTreeTransaction, PagedTargetState, TreeScratchArena


def make_requests(prefix_lengths=(256, 255, 257), num_blocks=32, device="cpu"):
    manager = BlockManager(num_blocks, 256)
    pool = torch.zeros((2, 3, num_blocks, 256, 2, 4), dtype=torch.bfloat16, device=device)
    states = []
    for request_id, length in enumerate(prefix_lengths):
        generator = torch.Generator().manual_seed(710 + request_id)
        kv = [(torch.randn((length, 2, 4), generator=generator).to(device=device, dtype=pool.dtype), torch.randn((length, 2, 4), generator=generator).to(device=device, dtype=pool.dtype)) for _ in range(3)]
        features = torch.full((1, length, 6), float(request_id), device=device)
        tokens = torch.full((1, length + 1), request_id, dtype=torch.long, device=device)
        states.append(PagedTargetState.from_prefill(tokens, kv, features, pool, manager, 256))
    arena = TreeScratchArena(pool, manager, 256)
    return states, arena, manager


def read_slots(pool, slots):
    return pool[:, :, slots // 256, slots % 256].clone()


def fill_tree(arena, nodes, request_id):
    generator = torch.Generator().manual_seed(910 + request_id)
    payload = torch.randn((2, 3, nodes.numel(), 2, 4), generator=generator).to(device=arena.kv_pool.device, dtype=arena.kv_pool.dtype)
    arena.kv_pool[:, :, nodes // 256, nodes % 256] = payload
    return payload


def commit_inputs(states, counts, paths):
    hidden = [torch.arange(count * 6, dtype=torch.float32, device=state.kv_pool.device).reshape(1, count, 6) + request_id * 10000 for request_id, (state, count) in enumerate(zip(states, counts))]
    tokens = [torch.cat((state.committed, torch.full((1, path.numel()), 900 + request_id, dtype=torch.long, device=state.kv_pool.device)), dim=1) for request_id, (state, path) in enumerate(zip(states, paths))]
    return hidden, tokens


class JetSpecBatchStateTest(unittest.TestCase):
    def cleanup(self, states, arena, manager):
        transaction = arena._batch_transaction
        if transaction is not None:
            transaction.abort()
        for state in states:
            state.clear()
        arena.clear()
        self.assertFalse(manager.used_block_ids)
        self.assertEqual(len(manager.free_block_ids), len(manager.blocks))
        self.assertTrue(all(block.ref_count == 0 for block in manager.blocks))

    def test_ragged_shared_page_and_one_all_layer_physical_copy(self):
        states, arena, manager = make_requests()
        prefixes = [read_slots(arena.kv_pool, state.logical_slots) for state in states]
        counts, maxima = [63, 31, 47], [4, 3, 5]
        paths = [torch.tensor([0, 1, 8, 15]), torch.tensor([0, 2, 30]), torch.tensor([0, 1, 7, 20, 46])]
        try:
            tx = BatchTreeTransaction.admit(states, counts, maxima, arena)
            self.assertEqual(tx.ranges, [(0, 63), (63, 94), (94, 141)])
            self.assertEqual(arena.capacity, 256)
            self.assertEqual(tx.packed_node_slots.numel(), 141)
            self.assertEqual(len(set(tx.packed_node_slots.tolist())), 141)
            self.assertTrue(all(not state.scratch_blocks for state in states))
            for state, nodes in zip(states, tx.node_slots):
                self.assertFalse(set(nodes.tolist()).intersection(state.logical_slots.tolist()))
            expected = [fill_tree(arena, nodes, request_id).index_select(2, path) for request_id, (nodes, path) in enumerate(zip(tx.node_slots, paths))]
            hidden, tokens = commit_inputs(states, counts, paths)
            original = state_module.copy_accepted_kv
            with patch.object(state_module, "copy_accepted_kv", wraps=original) as copy:
                metrics = tx.commit(hidden, paths, tokens)
            self.assertEqual(copy.call_count, 1)
            self.assertEqual(copy.call_args.args[1].numel(), 12)
            self.assertEqual(metrics["kv_copy_bytes"], 12 * 2 * 3 * 2 * 4 * 2)
            self.assertEqual(metrics["scratch_blocks"], 1)
            self.assertEqual(len(metrics["request_metrics"]), 3)
            self.assertFalse(tx.active)
            self.assertTrue(tx.committed)
            self.assertFalse(arena.active)
            self.assertIsNone(arena._batch_transaction)
            for state, path, source, prefix, features in zip(states, paths, expected, prefixes, hidden):
                state.assert_round_invariant()
                destination = read_slots(arena.kv_pool, state.logical_slots[-path.numel():])
                self.assertTrue(torch.equal(destination.view(torch.uint8), source.contiguous().view(torch.uint8)))
                self.assertTrue(torch.equal(read_slots(arena.kv_pool, state.logical_slots[:-path.numel()]), prefix))
                self.assertTrue(torch.equal(state.target_hidden[:, -path.numel():], features.index_select(1, path)))
                self.assertIsNot(state.scratch, arena)
            self.assertEqual(tx.abort(), 0)
            with self.assertRaisesRegex(RuntimeError, "no longer active"):
                tx.commit(hidden, paths, tokens)
        finally:
            self.cleanup(states, arena, manager)

    def test_c1_c2_c4_admission_and_independent_commit(self):
        for counts in ([63], [63, 31], [63, 31, 47, 63]):
            with self.subTest(concurrency=len(counts)):
                states, arena, manager = make_requests(prefix_lengths=tuple(253 + index for index in range(len(counts))))
                try:
                    tx = BatchTreeTransaction.admit(states, counts, [4] * len(counts), arena)
                    paths = [torch.tensor([0, index + 1]) for index in range(len(counts))]
                    expected = [fill_tree(arena, nodes, index).index_select(2, path) for index, (nodes, path) in enumerate(zip(tx.node_slots, paths))]
                    hidden, tokens = commit_inputs(states, counts, paths)
                    metrics = tx.commit(hidden, paths, tokens)
                    self.assertEqual(metrics["accepted_kv_slots"], 2 * len(counts))
                    self.assertEqual(len(arena.blocks), 1)
                    for state, raw in zip(states, expected):
                        state.assert_round_invariant()
                        self.assertTrue(torch.equal(read_slots(arena.kv_pool, state.logical_slots[-2:]), raw))
                finally:
                    self.cleanup(states, arena, manager)

    def test_arena_high_water_growth_then_smaller_batch_is_reused(self):
        states, arena, manager = make_requests()
        try:
            first = BatchTreeTransaction.admit(states, [63, 31, 47], [4, 4, 4], arena)
            first.abort()
            self.assertEqual(len(arena.blocks), 1)
            second = BatchTreeTransaction.admit(states, [200, 200, 200], [4, 4, 4], arena)
            self.assertEqual(len(arena.blocks), 3)
            second.abort()
            high_water = list(arena.blocks)
            for _ in range(10):
                BatchTreeTransaction.admit(states, [63, 31, 47], [4, 4, 4], arena).abort()
                self.assertEqual(arena.blocks, high_water)
            for state in states:
                state.clear()
            self.assertEqual(manager.used_block_ids, set(high_water))
        finally:
            self.cleanup(states, arena, manager)

    def test_request_cannot_retire_or_free_borrowed_active_scratch(self):
        states, arena, manager = make_requests()
        try:
            tx = BatchTreeTransaction.admit(states, [63, 31, 47], [4, 4, 4], arena)
            before = set(manager.used_block_ids)
            for state in states:
                for operation in (state.clear, state.abort_tree, lambda: state.reserve_tree(63)):
                    with self.assertRaisesRegex(RuntimeError, "batch transaction"):
                        operation()
            with self.assertRaisesRegex(RuntimeError, "batch transaction"):
                arena.clear()
            with self.assertRaisesRegex(RuntimeError, "already leased"):
                BatchTreeTransaction.admit(states, [63, 31, 47], [4, 4, 4], arena)
            self.assertEqual(before, manager.used_block_ids)
            self.assertTrue(tx.active)
            self.assertTrue(arena.active)
            tx.abort()
            for state in states:
                state.assert_round_invariant()
        finally:
            self.cleanup(states, arena, manager)

    def test_atomic_admission_failure_leaves_every_request_and_arena_unchanged(self):
        states, arena, manager = make_requests(num_blocks=6)
        before = set(manager.used_block_ids)
        try:
            with self.assertRaisesRegex(RuntimeError, "insufficient KV blocks"):
                BatchTreeTransaction.admit(states, [63, 31, 47], [4, 4, 4], arena)
            self.assertEqual(before, manager.used_block_ids)
            self.assertFalse(arena.blocks)
            self.assertFalse(arena.active)
            self.assertTrue(all(not state.pending_blocks and state._batch_transaction is None for state in states))
        finally:
            self.cleanup(states, arena, manager)

    def test_failed_admission_preserves_existing_arena_capacity(self):
        states, arena, manager = make_requests(num_blocks=8)
        held = []
        try:
            BatchTreeTransaction.admit(states, [63, 31, 47], [4, 4, 4], arena).abort()
            old_blocks = list(arena.blocks)
            held = manager.reserve_provisional(2)
            before = set(manager.used_block_ids)
            with self.assertRaisesRegex(RuntimeError, "insufficient KV blocks"):
                BatchTreeTransaction.admit(states, [200, 200, 200], [4, 4, 4], arena)
            self.assertEqual(before, manager.used_block_ids)
            self.assertEqual(old_blocks, arena.blocks)
        finally:
            manager.release_provisional(held)
            self.cleanup(states, arena, manager)

    def test_metadata_exception_rolls_back_new_arena_growth_and_destinations(self):
        states, arena, manager = make_requests()
        before = set(manager.used_block_ids)
        original = state_module._slots_for_blocks

        def fail_packed_metadata(blocks, length, *args, **kwargs):
            if length == 141:
                raise RuntimeError("injected packed metadata failure")
            return original(blocks, length, *args, **kwargs)

        try:
            with patch.object(state_module, "_slots_for_blocks", side_effect=fail_packed_metadata):
                with self.assertRaisesRegex(RuntimeError, "metadata failure"):
                    BatchTreeTransaction.admit(states, [63, 31, 47], [4, 4, 4], arena)
            self.assertEqual(before, manager.used_block_ids)
            self.assertFalse(arena.blocks)
            self.assertFalse(arena.active)
            for state in states:
                state.assert_round_invariant()
        finally:
            self.cleanup(states, arena, manager)

    def test_all_paths_prepared_before_copy_invalid_last_request_publishes_nothing(self):
        states, arena, manager = make_requests()
        try:
            tx = BatchTreeTransaction.admit(states, [63, 31, 47], [4, 4, 4], arena)
            paths = [torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([0, 47])]
            hidden, tokens = commit_inputs(states, [63, 31, 47], paths)
            original_tokens = [state.committed for state in states]
            with patch.object(state_module, "copy_accepted_kv") as copy:
                with self.assertRaisesRegex(ValueError, "bounds"):
                    tx.commit(hidden, paths, tokens)
            copy.assert_not_called()
            self.assertTrue(all(state.committed is tokens for state, tokens in zip(states, original_tokens)))
            self.assertTrue(tx.active)
            tx.abort()
        finally:
            self.cleanup(states, arena, manager)

    def test_partial_copy_failure_preserves_all_prefixes_until_batch_abort(self):
        for after_copy in (False, True):
            with self.subTest(after_copy=after_copy):
                states, arena, manager = make_requests()
                try:
                    prefixes = [read_slots(arena.kv_pool, state.logical_slots) for state in states]
                    previous = [(state.committed, state.target_hidden, state.logical_slots, list(state.owned_blocks)) for state in states]
                    tx = BatchTreeTransaction.admit(states, [63, 31, 47], [4, 4, 4], arena)
                    paths = [torch.tensor([0, 1, 8, 15]), torch.tensor([0, 1]), torch.tensor([0, 3, 40])]
                    for request_id, nodes in enumerate(tx.node_slots):
                        fill_tree(arena, nodes, request_id)
                    hidden, tokens = commit_inputs(states, [63, 31, 47], paths)
                    original = state_module.copy_accepted_kv

                    def fail_copy(*args):
                        if after_copy:
                            original(*args)
                        raise RuntimeError("injected accepted KV copy failure")

                    with patch.object(state_module, "copy_accepted_kv", side_effect=fail_copy):
                        with self.assertRaisesRegex(RuntimeError, "copy failure"):
                            tx.commit(hidden, paths, tokens)
                    self.assertFalse(tx.committed)
                    for state, old, prefix in zip(states, previous, prefixes):
                        self.assertIs(state.committed, old[0])
                        self.assertIs(state.target_hidden, old[1])
                        self.assertIs(state.logical_slots, old[2])
                        self.assertEqual(state.owned_blocks, old[3])
                        self.assertTrue(torch.equal(read_slots(arena.kv_pool, state.logical_slots), prefix))
                    self.assertEqual(tx.abort(), 2)
                    self.assertEqual(tx.abort(), 0)
                    self.assertEqual(len(arena.blocks), 1)
                    for state in states:
                        state.assert_round_invariant()
                finally:
                    self.cleanup(states, arena, manager)

    def test_interrupted_unused_page_release_rolls_back_publication_and_allocator(self):
        for interrupt_at in ("before", "middle", "after"):
            with self.subTest(interrupt_at=interrupt_at):
                states, arena, manager = make_requests(prefix_lengths=(253, 253, 253))
                try:
                    tx = BatchTreeTransaction.admit(states, [63, 31, 47], [4, 4, 4], arena)
                    before = set(manager.used_block_ids)
                    free_before = tuple(manager.free_block_ids)
                    original = manager._deallocate_block
                    paths = [torch.tensor([0])] * 3
                    hidden, tokens = commit_inputs(states, [63, 31, 47], paths)
                    previous = [(state.committed, state.logical_slots, state.target_hidden, list(state.owned_blocks), list(state.pending_blocks)) for state in states]

                    def interrupt_release(block_id):
                        if interrupt_at == "middle":
                            manager.used_block_ids.remove(block_id)
                        elif interrupt_at == "after":
                            original(block_id)
                        raise KeyboardInterrupt("injected allocator interruption")

                    with patch.object(manager, "_deallocate_block", side_effect=interrupt_release):
                        with self.assertRaisesRegex(KeyboardInterrupt, "allocator interruption"):
                            tx.commit(hidden, paths, tokens)
                    self.assertFalse(tx.committed)
                    self.assertEqual(before, manager.used_block_ids)
                    self.assertEqual(free_before, tuple(manager.free_block_ids))
                    self.assertFalse(set(manager.free_block_ids).intersection(before))
                    for state, old in zip(states, previous):
                        self.assertIs(state.committed, old[0])
                        self.assertIs(state.logical_slots, old[1])
                        self.assertIs(state.target_hidden, old[2])
                        self.assertEqual(state.owned_blocks, old[3])
                        self.assertEqual(state.pending_blocks, old[4])
                    self.assertEqual(tx.abort(), 3)
                    for state in states:
                        state.assert_round_invariant()
                finally:
                    self.cleanup(states, arena, manager)

    def test_interrupt_after_release_return_preserves_terminal_commit(self):
        # Exercise both actual unused-page release and the zero-unused-page case.
        for prompt_len in (253, 256):
            with self.subTest(prompt_len=prompt_len):
                states, arena, manager = make_requests(prefix_lengths=(prompt_len,) * 3)
                try:
                    tx = BatchTreeTransaction.admit(states, [63, 31, 47], [4, 4, 4], arena)
                    paths = [torch.tensor([0])] * 3
                    hidden, tokens = commit_inputs(states, [63, 31, 47], paths)
                    original = manager.release_provisional

                    def interrupt_after_release(block_ids):
                        original(block_ids)
                        raise KeyboardInterrupt("injected after completed release")

                    with patch.object(manager, "release_provisional", side_effect=interrupt_after_release):
                        with self.assertRaisesRegex(KeyboardInterrupt, "completed release"):
                            tx.commit(hidden, paths, tokens)
                    self.assertTrue(tx.committed)
                    self.assertFalse(tx.active)
                    self.assertIsNone(arena._batch_transaction)
                    self.assertEqual(tx.result["live_kv_slots"], 3 * (prompt_len + 1))
                    for state, expected_tokens in zip(states, tokens):
                        self.assertIs(state.committed, expected_tokens)
                        self.assertFalse(state.pending_blocks)
                        state.assert_round_invariant()
                    self.assertEqual(tx.abort(), 0)
                finally:
                    self.cleanup(states, arena, manager)

    def test_interrupt_during_terminal_guard_cleanup_is_recoverable(self):
        states, arena, manager = make_requests()
        try:
            tx = BatchTreeTransaction.admit(states, [63, 31, 47], [4, 4, 4], arena)
            paths = [torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([0, 3])]
            hidden, tokens = commit_inputs(states, [63, 31, 47], paths)

            def interrupt_during_finish():
                # This is the production ordering: readiness is published before
                # an individual request's guard can be removed.
                states[0].scratch._retired = arena._retired
                states[0]._batch_transaction = None
                raise KeyboardInterrupt("injected during committed guard cleanup")

            with patch.object(tx, "finish_committed", side_effect=interrupt_during_finish):
                with self.assertRaisesRegex(KeyboardInterrupt, "guard cleanup"):
                    tx.commit(hidden, paths, tokens)
            self.assertTrue(tx.committed)
            self.assertTrue(tx.active)
            self.assertIsNone(states[0]._batch_transaction)
            self.assertIs(states[1]._batch_transaction, tx)
            self.assertIs(arena._batch_transaction, tx)
            with self.assertRaisesRegex(RuntimeError, "no longer active"):
                tx.commit(hidden, paths, tokens)
            self.assertEqual(tx.abort(), 0)
            self.assertFalse(tx.active)
            for state, expected_tokens in zip(states, tokens):
                self.assertIs(state.committed, expected_tokens)
                state.assert_round_invariant()
            self.assertEqual(tx.abort(), 0)
            tx.finish_committed()
        finally:
            self.cleanup(states, arena, manager)

    def test_finish_one_request_reorder_remaining_and_poison_reuse(self):
        states, arena, manager = make_requests()
        try:
            counts = [63, 31, 47]
            paths = [torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([0, 3])]
            tx = BatchTreeTransaction.admit(states, counts, [4, 4, 4], arena)
            for request_id, nodes in enumerate(tx.node_slots):
                fill_tree(arena, nodes, request_id)
            hidden, tokens = commit_inputs(states, counts, paths)
            tx.commit(hidden, paths, tokens)
            arena_blocks = list(arena.blocks)
            remaining = [states[2], states[0]]
            prefixes = [read_slots(arena.kv_pool, state.logical_slots) for state in remaining]
            states[1].clear()
            self.assertEqual(arena_blocks, arena.blocks)
            tx2 = BatchTreeTransaction.admit(remaining, [47, 63], [4, 4], arena)
            poison = torch.full((2, 3, 110, 2, 4), float("nan"), dtype=arena.kv_pool.dtype)
            arena.kv_pool[:, :, tx2.packed_node_slots // 256, tx2.packed_node_slots % 256] = poison
            self.assertEqual(arena_blocks, arena.blocks)
            for state, prefix in zip(remaining, prefixes):
                self.assertTrue(torch.equal(read_slots(arena.kv_pool, state.logical_slots), prefix))
            tx2.abort()
            for state in remaining:
                state.clear()
            self.assertEqual(manager.used_block_ids, set(arena_blocks))
            self.assertEqual(arena.clear(), 1)
            self.assertEqual(arena.clear(), 0)
        finally:
            self.cleanup(states, arena, manager)

    def test_1000_rounds_shared_scratch_not_historical_pages(self):
        path = torch.tensor([0, 1, 8, 15])
        for counts in ([63], [63, 31], [63, 31, 47], [63, 31, 47, 63]):
            concurrency = len(counts)
            final_blocks = 17 * concurrency + 1
            with self.subTest(concurrency=concurrency, budgets=counts):
                # Size the real allocator to exactly final canonical pages plus
                # one shared scratch page. Per-round pinning cannot fit this pool.
                states, arena, manager = make_requests(prefix_lengths=(256,) * concurrency, num_blocks=final_blocks)
                initial_arena = None
                try:
                    for _ in range(1000):
                        tx = BatchTreeTransaction.admit(states, counts, [16] * concurrency, arena)
                        if initial_arena is None:
                            initial_arena = list(arena.blocks)
                        self.assertEqual(initial_arena, arena.blocks)
                        hidden, tokens = commit_inputs(states, counts, [path] * concurrency)
                        tx.commit(hidden, [path] * concurrency, tokens)
                    self.assertEqual([state.cache_len for state in states], [4256] * concurrency)
                    self.assertEqual([len(state.owned_blocks) for state in states], [17] * concurrency)
                    self.assertEqual(len(arena.blocks), 1)
                    self.assertEqual(len(manager.used_block_ids), final_blocks)
                    capacity = tx.capacity_snapshot()
                    self.assertEqual(capacity["reserved_slots"], final_blocks * 256)
                    self.assertEqual(capacity["live_slots"], 4256 * concurrency)
                    self.assertAlmostEqual(capacity["amplification"], final_blocks * 256 / (4256 * concurrency))
                    for state in states:
                        state.assert_round_invariant()
                finally:
                    self.cleanup(states, arena, manager)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class JetSpecBatchStreamTest(unittest.TestCase):
    cleanup = JetSpecBatchStateTest.cleanup

    def tearDown(self):
        torch.cuda.synchronize()

    def delay(self):
        if hasattr(torch.cuda, "_sleep"):
            torch.cuda._sleep(100_000_000)

    def assert_in_flight(self, event):
        if hasattr(torch.cuda, "_sleep"):
            self.assertFalse(event.query(), "handoff did not exercise outstanding CUDA work")

    def test_shared_copy_event_protects_reuse_by_new_requests_on_another_stream(self):
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            all_states, arena, manager = make_requests(prefix_lengths=(256, 255, 257, 31, 29), device="cuda")
            states = all_states[:3]
            replacements = all_states[3:]
            counts = [63, 31, 47]
            tx = BatchTreeTransaction.admit(states, counts, [4, 4, 4], arena)
            paths = [torch.tensor([0, 1, 8, 15], device="cuda"), torch.tensor([0, 2], device="cuda"), torch.tensor([0, 3, 40], device="cuda")]
            expected = [fill_tree(arena, nodes, index).index_select(2, path).clone() for index, (nodes, path) in enumerate(zip(tx.node_slots, paths))]
            hidden, tokens = commit_inputs(states, counts, paths)
            original = state_module.copy_accepted_kv

            def delayed_copy(*args):
                self.delay()
                return original(*args)

            with patch.object(state_module, "copy_accepted_kv", side_effect=delayed_copy):
                tx.commit(hidden, paths, tokens)
            self.assert_in_flight(arena._retired)
        try:
            with torch.cuda.stream(consumer):
                # These new requests' prefill events predate the delayed copy.
                # Only the runner arena dependency can protect scratch reuse.
                reused = BatchTreeTransaction.admit(replacements, [63, 31], [4, 4], arena)
                poison = torch.full((2, 3, 94, 2, 4), float("nan"), dtype=arena.kv_pool.dtype, device="cuda")
                arena.kv_pool[:, :, reused.packed_node_slots // 256, reused.packed_node_slots % 256] = poison
                for state, path, raw in zip(states, paths, expected):
                    self.assertTrue(torch.equal(read_slots(arena.kv_pool, state.logical_slots[-path.numel():]), raw))
                reused.abort()
                all_states[1].clear()
                self.assertEqual(len(arena.blocks), 1)
        finally:
            with torch.cuda.stream(consumer):
                self.cleanup(all_states, arena, manager)

    def test_shared_partial_copy_failure_abort_fences_all_destinations(self):
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            states, arena, manager = make_requests(device="cuda")
            prefixes = [read_slots(arena.kv_pool, state.logical_slots) for state in states]
            counts = [63, 31, 47]
            tx = BatchTreeTransaction.admit(states, counts, [4, 4, 4], arena)
            paths = [torch.tensor([0, 1, 8, 15], device="cuda"), torch.tensor([0, 2], device="cuda"), torch.tensor([0, 3, 40], device="cuda")]
            for index, nodes in enumerate(tx.node_slots):
                fill_tree(arena, nodes, index)
            hidden, tokens = commit_inputs(states, counts, paths)
            original = state_module.copy_accepted_kv

            def fail_after_copy(*args):
                self.delay()
                original(*args)
                raise RuntimeError("injected shared asynchronous copy failure")

            with patch.object(state_module, "copy_accepted_kv", side_effect=fail_after_copy):
                with self.assertRaisesRegex(RuntimeError, "copy failure"):
                    tx.commit(hidden, paths, tokens)
            completion = torch.cuda.Event()
            completion.record(producer)
            self.assert_in_flight(completion)
        try:
            with torch.cuda.stream(consumer):
                self.assertEqual(tx.abort(), 2)
                self.assertTrue(completion.query())
                self.assertTrue(arena._retired.query())
                for state, prefix in zip(states, prefixes):
                    state.assert_round_invariant()
                    self.assertTrue(torch.equal(read_slots(arena.kv_pool, state.logical_slots), prefix))
        finally:
            with torch.cuda.stream(consumer):
                self.cleanup(states, arena, manager)

    def test_shared_verify_abort_without_destinations_defers_reuse_until_event(self):
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            all_states, arena, manager = make_requests(prefix_lengths=(64, 65, 31, 29), device="cuda")
            tx = BatchTreeTransaction.admit(all_states[:2], [63, 31], [4, 4], arena)
            self.assertTrue(all(not state.pending_blocks for state in tx.states))
            old_payload = torch.ones((2, 3, 94, 2, 4), dtype=arena.kv_pool.dtype, device="cuda")
            self.delay()
            arena.kv_pool[:, :, tx.packed_node_slots // 256, tx.packed_node_slots % 256] = old_payload
        try:
            with torch.cuda.stream(consumer):
                self.assertEqual(tx.abort(), 0)
                self.assert_in_flight(arena._retired)
                reused = BatchTreeTransaction.admit(all_states[2:], [63, 31], [4, 4], arena)
                new_payload = torch.full((2, 3, 94, 2, 4), 3.0, dtype=arena.kv_pool.dtype, device="cuda")
                arena.kv_pool[:, :, reused.packed_node_slots // 256, reused.packed_node_slots % 256] = new_payload
                self.assertTrue(torch.equal(read_slots(arena.kv_pool, reused.packed_node_slots), new_payload))
                reused.abort()
        finally:
            with torch.cuda.stream(consumer):
                self.cleanup(all_states, arena, manager)


if __name__ == "__main__":
    unittest.main()
