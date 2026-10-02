"""CPU correctness gates for canonical JetSpec KV ownership.

Run with ``python -m unittest discover -s tests -v``.  No model, CUDA
device, or pytest dependency is needed.  ``--capacity-json`` also emits
the 10/100/1000-round allocator replay metrics used by the Phase 3 report.
These are allocator/state tests, not attention-kernel numerical tests.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.engine.block_manager import BlockManager
from nanovllm.speculative.jetspec import state as state_module
from nanovllm.speculative.jetspec.state import PagedTargetState


def make_state(
    prompt_len: int = 256,
    *,
    num_blocks: int = 24,
    block_size: int = 256,
    layers: int = 3,
    dtype: torch.dtype = torch.float32,
    held_blocks: int = 0,
) -> tuple[PagedTargetState, BlockManager, torch.Tensor, list[int]]:
    manager = BlockManager(num_blocks, block_size)
    held = manager.reserve_provisional(held_blocks)
    pool = torch.zeros((2, layers, num_blocks, block_size, 2, 4), dtype=dtype)
    generator = torch.Generator().manual_seed(173)
    prompt_kv = [
        (
            torch.randn((prompt_len, 2, 4), generator=generator).to(dtype),
            torch.randn((prompt_len, 2, 4), generator=generator).to(dtype),
        )
        for _ in range(layers)
    ]
    features = torch.arange(prompt_len * 6, dtype=torch.float32).reshape(1, prompt_len, 6)
    committed = torch.arange(prompt_len + 1, dtype=torch.long).reshape(1, -1)
    state = PagedTargetState.from_prefill(
        committed, prompt_kv, features, pool, manager, block_size
    )
    state.assert_round_invariant()
    return state, manager, pool, held


def read_slots(pool: torch.Tensor, slots: torch.Tensor, block_size: int) -> torch.Tensor:
    return pool[:, :, slots // block_size, slots % block_size].clone()


def write_tree(state: PagedTargetState, slots: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (2, state.kv_pool.shape[1], slots.numel(), *state.kv_pool.shape[-2:])
    values = torch.randn(shape, generator=generator).to(state.kv_pool.dtype)
    state.kv_pool[:, :, slots // state.block_size, slots % state.block_size] = values
    return values


def next_tokens(state: PagedTargetState, growth: int) -> torch.Tensor:
    # Appended outputs are accepted draft tokens plus the uncached correction.
    return torch.cat((state.committed, torch.full((1, growth), 7, dtype=torch.long)), dim=1)


def capacity_replay(rounds: int) -> dict[str, int | float]:
    prompt_len, tree_nodes, growth, block_size = 256, 63, 4, 256
    final_live = prompt_len + rounds * growth
    # Exactly canonical final capacity plus one scratch page: historical pinning
    # would exhaust this real allocator far before the 1000-round gate.
    total_blocks = math.ceil(final_live / block_size) + 1
    state, manager, pool, _ = make_state(num_blocks=total_blocks)
    path = torch.tensor([0, 1, 8, 15])
    first_scratch = None
    max_used_blocks = len(manager.used_block_ids)
    copied_bytes = 0
    try:
        for round_id in range(rounds):
            nodes, visible = state.reserve_tree(tree_nodes, max_path_length=growth)
            assert torch.equal(visible[:state.cache_len], state.logical_slots)
            assert not set(nodes.tolist()).intersection(state.logical_slots.tolist())
            scratch_ids = tuple(state.scratch_blocks)
            if first_scratch is None:
                first_scratch = scratch_ids
            assert scratch_ids == first_scratch, "scratch allocation grew or changed across rounds"
            max_used_blocks = max(max_used_blocks, len(manager.used_block_ids))
            write_tree(state, nodes, seed=1000 + round_id)
            hidden = torch.full((1, tree_nodes, 6), float(round_id))
            metrics = state.commit_tree_path(
                nodes, hidden, path, committed_tokens=next_tokens(state, growth)
            )
            copied_bytes += metrics["kv_copy_bytes"]
            state.assert_round_invariant()
            assert len(state.owned_blocks) == math.ceil(state.cache_len / block_size)
            assert len(state.scratch_blocks) == 1
            assert not state.pending_blocks
            assert not state.scratch_active
            assert len(manager.used_block_ids) == len(state.owned_blocks) + 1
        bytes_per_token = 2 * pool.shape[1] * pool.shape[-2] * pool.shape[-1] * pool.element_size()
        assert copied_bytes == rounds * growth * bytes_per_token, "copy accounting depends on prefix history"
        result = {
            "rounds": rounds,
            "prompt_slots": prompt_len,
            "tree_nodes": tree_nodes,
            "accepted_root_inclusive": growth,
            "live_slots": state.cache_len,
            "committed_blocks": len(state.owned_blocks),
            "committed_capacity_slots": len(state.owned_blocks) * block_size,
            "scratch_blocks": len(state.scratch_blocks),
            "scratch_capacity_slots": len(state.scratch_blocks) * block_size,
            "reserved_blocks": len(manager.used_block_ids),
            "reserved_capacity_slots": len(manager.used_block_ids) * block_size,
            "peak_used_blocks": max_used_blocks,
            "kv_copy_payload_bytes": copied_bytes,
            "amplification_including_scratch": len(manager.used_block_ids) * block_size / state.cache_len,
            "old_baseline_reserved_capacity_slots": (1 + rounds) * block_size,
        }
    finally:
        state.clear()
    assert not manager.used_block_ids
    assert len(manager.free_block_ids) == total_blocks
    assert state.clear() == 0
    return result


class JetSpecKVLifetimeTest(unittest.TestCase):
    def assert_raw_equal(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                    expected.contiguous().view(torch.uint8)),
                        "K/V byte payload changed during accepted-only commit")

    def assert_allocator_restored(self, manager: BlockManager, held: list[int] = ()) -> None:
        self.assertEqual(manager.used_block_ids, set(held))
        free = list(manager.free_block_ids)
        self.assertEqual(len(free), len(set(free)), "free list contains duplicate page IDs")
        self.assertEqual(set(free) | set(held), set(range(len(manager.blocks))))
        for block in manager.blocks:
            self.assertEqual(block.ref_count, int(block.block_id in held))

    def test_capacity_10_100_1000_rounds(self) -> None:
        expected = {10: (296, 512, 768), 100: (656, 768, 1024), 1000: (4256, 4352, 4608)}
        for rounds, (live, committed, total) in expected.items():
            with self.subTest(rounds=rounds):
                result = capacity_replay(rounds)
                self.assertEqual(result["live_slots"], live)
                self.assertEqual(result["committed_capacity_slots"], committed)
                self.assertEqual(result["reserved_capacity_slots"], total)
                self.assertEqual(result["scratch_blocks"], 1)

    def test_raw_bytes_all_layers_and_canonical_noncontiguous_pages(self) -> None:
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                state, manager, pool, held = make_state(prompt_len=255, dtype=dtype, held_blocks=2)
                # Occupied pages guarantee a nonzero physical prefix and a gap
                # between its first page and its next canonical tail page.
                held.extend(manager.reserve_provisional(1))
                old_slots = state.logical_slots.clone()
                old_payload = read_slots(pool, old_slots, state.block_size)
                nodes, _ = state.reserve_tree(63, max_path_length=16)
                node_payload = write_tree(state, nodes, seed=923)
                path = torch.tensor([0, 1, 8, 15])
                hidden = torch.arange(63 * 6, dtype=torch.float32).reshape(1, 63, 6)
                expected_features = torch.cat((state.target_hidden, hidden[:, path]), dim=1)
                full_tokens = next_tokens(state, path.numel())
                metrics = state.commit_tree_path(nodes, hidden, path, committed_tokens=full_tokens)
                destinations = state.logical_slots[-path.numel():]
                self.assertFalse(set(destinations.tolist()).intersection(nodes.tolist()))
                self.assert_raw_equal(read_slots(pool, destinations, state.block_size),
                                      node_payload.index_select(2, path))
                self.assert_raw_equal(read_slots(pool, old_slots, state.block_size), old_payload)
                self.assertTrue(torch.equal(state.target_hidden, expected_features))
                self.assertTrue(torch.equal(state.committed, full_tokens))
                self.assertEqual(metrics["kv_copy_bytes"],
                                 path.numel() * 2 * 3 * 2 * 4 * pool.element_size())
                positions = torch.arange(state.cache_len)
                table = torch.tensor(state.owned_blocks)
                self.assertTrue(torch.equal(state.logical_slots,
                    table[positions // state.block_size] * state.block_size + positions % state.block_size))
                self.assertNotEqual(state.owned_blocks[1], state.owned_blocks[0] + 1)
                state.assert_round_invariant()
                for block_id in state.scratch_blocks:
                    self.assertEqual(manager.blocks[block_id].hash, -1)
                    self.assertNotIn(block_id, manager.hash_to_block_id.values())
                state.clear()
                self.assertEqual(state.clear(), 0)
                self.assert_allocator_restored(manager, held)
                manager.release_provisional(held)
                self.assert_allocator_restored(manager)

    def test_rejected_poison_and_reused_scratch_cannot_change_history(self) -> None:
        state, manager, pool, _ = make_state(dtype=torch.bfloat16)
        path = torch.tensor([0, 1, 8, 15])
        nodes, _ = state.reserve_tree(63, max_path_length=4)
        initial_tree = write_tree(state, nodes, seed=871)
        state.commit_tree_path(nodes, torch.zeros(1, 63, 6), path,
                               committed_tokens=next_tokens(state, 4))
        committed_before_reuse = read_slots(pool, state.logical_slots, state.block_size)
        committed_ids = set(state.logical_slots.tolist())
        self.assertFalse(committed_ids.intersection(nodes.tolist()))
        # Poison *all* old scratch slots, including old accepted source rows.
        # Accepted source rows are no longer canonical ownership after copying.
        pool[:, :, nodes // state.block_size, nodes % state.block_size] = float("nan")
        self.assert_raw_equal(read_slots(pool, state.logical_slots, state.block_size),
                              committed_before_reuse)
        next_nodes, next_visible = state.reserve_tree(63, max_path_length=4)
        self.assertTrue(torch.equal(nodes, next_nodes), "scratch was not reused")
        prefix = next_visible[:state.cache_len]
        self.assertTrue(torch.equal(prefix, state.logical_slots))
        self.assertFalse(set(prefix.tolist()).intersection(nodes.tolist()))
        new_tree = write_tree(state, next_nodes, seed=872)
        self.assertFalse(torch.isnan(read_slots(pool, next_visible, state.block_size)).any())
        state.commit_tree_path(next_nodes, torch.ones(1, 63, 6), path,
                               committed_tokens=next_tokens(state, 4))
        self.assert_raw_equal(read_slots(pool, state.logical_slots[:-4], state.block_size),
                              committed_before_reuse)
        self.assert_raw_equal(read_slots(pool, state.logical_slots[-4:], state.block_size),
                              new_tree.index_select(2, path))
        self.assertFalse(torch.equal(initial_tree, new_tree))
        state.assert_round_invariant()
        state.clear()
        self.assert_allocator_restored(manager)

    def test_root_only_returns_unneeded_destination_reservation(self) -> None:
        state, manager, _, _ = make_state(prompt_len=255, num_blocks=4)
        nodes, _ = state.reserve_tree(63, max_path_length=16)
        self.assertEqual(len(state.pending_blocks), 1)
        self.assertEqual(len(manager.used_block_ids), 3)
        write_tree(state, nodes, seed=43)
        state.commit_tree_path(nodes, torch.ones(1, 63, 6), torch.tensor([0]),
                               committed_tokens=next_tokens(state, 1))
        self.assertEqual(state.cache_len, 256)
        self.assertEqual(len(state.owned_blocks), 1)
        self.assertEqual(len(state.scratch_blocks), 1)
        self.assertEqual(len(manager.used_block_ids), 2)
        state.assert_round_invariant()
        state.clear()
        self.assert_allocator_restored(manager)

    def test_long_path_crosses_canonical_page_and_multiple_scratch_pages(self) -> None:
        # Parameterized state geometry, not a change to the serving page size.
        state, manager, pool, _ = make_state(prompt_len=15, block_size=16, num_blocks=12)
        nodes, _ = state.reserve_tree(63, max_path_length=16)
        self.assertEqual(len(state.scratch_blocks), 4)
        payload = write_tree(state, nodes, seed=233)
        path = torch.tensor([0, 17, 35, 62])
        state.commit_tree_path(nodes, torch.ones(1, 63, 6), path,
                               committed_tokens=next_tokens(state, 4))
        self.assert_raw_equal(read_slots(pool, state.logical_slots[-4:], 16),
                              payload.index_select(2, path))
        state.assert_round_invariant()
        self.assertEqual(len(state.owned_blocks), 2)
        self.assertEqual(len(manager.used_block_ids), 6)
        state.clear()
        self.assert_allocator_restored(manager)

    def test_admission_failure_is_atomic_before_verify(self) -> None:
        # One free page can hold scratch OR destination, but cannot admit both.
        state, manager, pool, _ = make_state(num_blocks=2)
        old_payload = read_slots(pool, state.logical_slots, state.block_size)
        old_used = manager.used_block_ids.copy()
        with self.assertRaisesRegex(RuntimeError, "insufficient"):
            state.reserve_tree(63, max_path_length=4)
        self.assertEqual(manager.used_block_ids, old_used)
        self.assertFalse(state.pending_blocks)
        self.assertFalse(state.scratch_blocks)
        self.assertFalse(state.scratch_active)
        self.assert_raw_equal(read_slots(pool, state.logical_slots, state.block_size), old_payload)
        state.assert_round_invariant()
        state.clear()
        self.assert_allocator_restored(manager)

    def test_capacity_failure_with_existing_scratch_preserves_arena(self) -> None:
        state, manager, _, _ = make_state(prompt_len=255, num_blocks=3)
        nodes, _ = state.reserve_tree(63, max_path_length=1)
        write_tree(state, nodes, seed=552)
        state.commit_tree_path(nodes, torch.ones(1, 63, 6), torch.tensor([0]),
                               committed_tokens=next_tokens(state, 1))
        scratch_before = state.scratch_blocks.copy()
        held = manager.reserve_provisional(1)  # Pool is now full.
        used_before = manager.used_block_ids.copy()
        with self.assertRaisesRegex(RuntimeError, "insufficient"):
            state.reserve_tree(63, max_path_length=4)
        self.assertEqual(manager.used_block_ids, used_before)
        self.assertEqual(state.scratch_blocks, scratch_before)
        self.assertFalse(state.pending_blocks)
        self.assertFalse(state.scratch_active)
        state.assert_round_invariant()
        manager.release_provisional(held)
        nodes, _ = state.reserve_tree(63, max_path_length=4)
        state.abort_tree()
        state.clear()
        self.assert_allocator_restored(manager)

    def test_allocator_injected_partial_failure_rolls_back(self) -> None:
        manager = BlockManager(5, 256)
        held = manager.reserve_provisional(1)
        real_allocate = manager._allocate_block
        calls = 0

        def failing_allocate():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected page allocation failure")
            return real_allocate()

        with patch.object(manager, "_allocate_block", side_effect=failing_allocate):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                manager.reserve_provisional(3)
        self.assert_allocator_restored(manager, held)
        manager.release_provisional(held)
        self.assert_allocator_restored(manager)

    def test_release_validates_whole_lease_before_mutating(self) -> None:
        manager = BlockManager(5, 256)
        owned = manager.reserve_provisional(2)
        for bad_lease in ([owned[0], 4], [owned[0], owned[0]]):
            with self.subTest(lease=bad_lease):
                with self.assertRaises((ValueError, RuntimeError)):
                    manager.release_provisional(bad_lease)
                self.assertEqual(manager.used_block_ids, set(owned))
                for block_id in owned:
                    self.assertEqual(manager.blocks[block_id].ref_count, 1)
        manager.release_provisional(owned)
        self.assert_allocator_restored(manager)

    def test_abort_retires_round_and_keeps_committed_history(self) -> None:
        state, manager, pool, _ = make_state(prompt_len=255)
        old_slots = state.logical_slots.clone()
        old_payload = read_slots(pool, old_slots, state.block_size)
        nodes, _ = state.reserve_tree(63, max_path_length=16)
        write_tree(state, nodes, seed=222)
        state.abort_tree()
        state.abort_tree()
        self.assertFalse(state.pending_blocks)
        self.assertFalse(state.scratch_active)
        self.assertTrue(torch.equal(state.logical_slots, old_slots))
        self.assert_raw_equal(read_slots(pool, old_slots, state.block_size), old_payload)
        state.assert_round_invariant()
        next_nodes, _ = state.reserve_tree(63, max_path_length=16)
        self.assertTrue(torch.equal(nodes, next_nodes))
        state.clear()  # Clear also covers an active, uncommitted round.
        self.assertEqual(state.clear(), 0)
        self.assert_allocator_restored(manager)

    def test_second_reservation_is_rejected_without_losing_active_lease(self) -> None:
        state, manager, _, _ = make_state()
        nodes, _ = state.reserve_tree(63, max_path_length=4)
        used_before = manager.used_block_ids.copy()
        pending_before = state.pending_blocks.copy()
        with self.assertRaises(RuntimeError):
            state.reserve_tree(63, max_path_length=4)
        self.assertEqual(manager.used_block_ids, used_before)
        self.assertEqual(state.pending_blocks, pending_before)
        self.assertTrue(state.scratch_active)
        state.abort_tree()
        state.assert_round_invariant()
        reused, _ = state.reserve_tree(63, max_path_length=4)
        self.assertTrue(torch.equal(nodes, reused))
        state.clear()
        self.assert_allocator_restored(manager)

    def test_invalid_token_publication_is_rejected_before_kv_copy(self) -> None:
        state, manager, pool, _ = make_state()
        old_payload = read_slots(pool, state.logical_slots, state.block_size)
        old_length = state.cache_len
        nodes, _ = state.reserve_tree(63, max_path_length=4)
        write_tree(state, nodes, seed=577)
        with patch.object(state_module, "copy_accepted_kv") as copy:
            with self.assertRaises((RuntimeError, ValueError)):
                state.commit_tree_path(nodes, torch.ones(1, 63, 6), torch.tensor([0, 1, 8, 15]),
                                       committed_tokens=next_tokens(state, 2))
            copy.assert_not_called()
        self.assertEqual(state.cache_len, old_length)
        self.assert_raw_equal(read_slots(pool, state.logical_slots, state.block_size), old_payload)
        state.abort_tree()
        state.assert_round_invariant()
        state.clear()
        self.assert_allocator_restored(manager)

    def test_invalid_paths_do_not_copy_or_publish(self) -> None:
        state, manager, _, _ = make_state()
        nodes, _ = state.reserve_tree(63, max_path_length=4)
        write_tree(state, nodes, seed=919)
        old_length = state.cache_len
        for invalid in ([], [1], [0, 1, 1], [0, 63], [0, -1], [0, 1, 2, 3, 4]):
            with self.subTest(path=invalid):
                with patch.object(state_module, "copy_accepted_kv") as copy:
                    with self.assertRaises(ValueError):
                        state.commit_tree_path(nodes, torch.ones(1, 63, 6),
                                               torch.tensor(invalid, dtype=torch.long))
                    copy.assert_not_called()
                self.assertEqual(state.cache_len, old_length)
                self.assertTrue(state.scratch_active)
        state.abort_tree()
        state.assert_round_invariant()
        state.clear()
        self.assert_allocator_restored(manager)

    def test_reservation_metadata_exception_rolls_back_both_leases(self) -> None:
        state, manager, _, _ = make_state()
        old_used = manager.used_block_ids.copy()
        with patch.object(state_module.torch, "cat", side_effect=RuntimeError("injected metadata failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                state.reserve_tree(63, max_path_length=4)
        self.assertEqual(manager.used_block_ids, old_used)
        self.assertFalse(state.pending_blocks)
        self.assertFalse(state.scratch_blocks)
        self.assertFalse(state.scratch_active)
        state.assert_round_invariant()
        state.reserve_tree(63, max_path_length=4)
        state.clear()
        self.assert_allocator_restored(manager)

    def test_copy_exception_never_publishes_partial_state(self) -> None:
        for after_copy in (False, True):
            with self.subTest(after_copy=after_copy):
                state, manager, pool, _ = make_state(prompt_len=255)
                old_slots = state.logical_slots.clone()
                old_features = state.target_hidden.clone()
                old_tokens = state.committed.clone()
                old_payload = read_slots(pool, old_slots, state.block_size)
                nodes, _ = state.reserve_tree(63, max_path_length=4)
                write_tree(state, nodes, seed=823)
                real_copy = state_module.copy_accepted_kv

                def failing_copy(*args, **kwargs):
                    if after_copy:
                        real_copy(*args, **kwargs)
                    raise RuntimeError("injected accepted-copy failure")

                with patch.object(state_module, "copy_accepted_kv", side_effect=failing_copy):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        state.commit_tree_path(nodes, torch.ones(1, 63, 6), torch.tensor([0, 1, 8, 15]),
                                               committed_tokens=next_tokens(state, 4))
                self.assertTrue(torch.equal(state.logical_slots, old_slots))
                self.assertTrue(torch.equal(state.target_hidden, old_features))
                self.assertTrue(torch.equal(state.committed, old_tokens))
                self.assert_raw_equal(read_slots(pool, old_slots, state.block_size), old_payload)
                state.abort_tree()
                state.assert_round_invariant()
                self.assertEqual(len(manager.used_block_ids), len(state.owned_blocks) + len(state.scratch_blocks))
                state.clear()
                self.assert_allocator_restored(manager)

    def test_prefill_copy_exception_releases_reserved_pages(self) -> None:
        manager = BlockManager(4, 256)
        held = manager.reserve_provisional(1)
        pool = torch.zeros(2, 2, 4, 256, 2, 4)
        prompt_kv = [(torch.zeros(10, 2, 4), torch.zeros(10, 2, 4)),
                     (torch.zeros(9, 2, 4), torch.zeros(9, 2, 4))]
        with self.assertRaises((ValueError, RuntimeError)):
            PagedTargetState.from_prefill(torch.zeros(1, 11, dtype=torch.long), prompt_kv,
                                          torch.zeros(1, 10, 6), pool, manager, 256)
        self.assert_allocator_restored(manager, held)
        manager.release_provisional(held)
        self.assert_allocator_restored(manager)

    def test_prefill_partial_scatter_exception_releases_reserved_pages(self) -> None:
        class FailingScatterPool(torch.Tensor):
            writes = 0

            def __setitem__(self, index, value):
                type(self).writes += 1
                if type(self).writes == 3:
                    raise RuntimeError("injected prefill scatter failure")
                return super().__setitem__(index, value)

        manager = BlockManager(4, 256)
        held = manager.reserve_provisional(1)
        pool = torch.zeros(2, 2, 4, 256, 2, 4).as_subclass(FailingScatterPool)
        prompt_kv = [(torch.ones(10, 2, 4), torch.ones(10, 2, 4)) for _ in range(2)]
        with self.assertRaisesRegex(RuntimeError, "injected"):
            PagedTargetState.from_prefill(torch.zeros(1, 11, dtype=torch.long), prompt_kv,
                                          torch.zeros(1, 10, 6), pool, manager, 256)
        self.assertEqual(FailingScatterPool.writes, 3)
        self.assertTrue(torch.count_nonzero(pool).item(), "test did not queue partial prefill writes")
        self.assert_allocator_restored(manager, held)
        manager.release_provisional(held)
        self.assert_allocator_restored(manager)

    def test_request_exception_finally_releases_active_round(self) -> None:
        state, manager, _, _ = make_state()
        with self.assertRaisesRegex(RuntimeError, "verification failure"):
            try:
                nodes, _ = state.reserve_tree(63, max_path_length=4)
                write_tree(state, nodes, seed=616)
                raise RuntimeError("injected verification failure")
            finally:
                state.clear()
        self.assertEqual(state.clear(), 0)
        self.assert_allocator_restored(manager)

    def test_runtime_public_wrapper_finally_covers_errors_and_cancellation(self) -> None:
        from nanovllm.speculative.jetspec import runtime as runtime_module

        for error_class in (RuntimeError, KeyboardInterrupt):
            for method, inner_method in (("generate", "_generate_request"),
                                         ("generate_target_paged", "_generate_target_paged_request")):
                with self.subTest(error=error_class.__name__, method=method):
                    state, manager, _, _ = make_state()
                    runtime = runtime_module.JetSpecRuntime.__new__(runtime_module.JetSpecRuntime)
                    runtime._active_state = None
                    runtime.drafter = Mock()

                    def fail_request(*args, **kwargs):
                        runtime._active_state = state
                        nodes, _ = state.reserve_tree(63, max_path_length=4)
                        write_tree(state, nodes, seed=932)
                        raise error_class("injected request interruption")

                    setattr(runtime, inner_method, fail_request)
                    with patch.object(runtime_module, "reset_context") as reset_context:
                        with self.assertRaisesRegex(error_class, "injected"):
                            getattr(runtime, method)([1, 2, 3])
                        reset_context.assert_called()
                    self.assertIsNone(runtime._active_state)
                    if method == "generate":
                        runtime.drafter.reset_cache.assert_called_once()
                    self.assertEqual(state.clear(), 0)
                    self.assert_allocator_restored(manager)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-json", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.capacity_json:
        print(json.dumps({"capacity_replays": [capacity_replay(n) for n in (10, 100, 1000)]}, indent=2))
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
