"""Optional real-CUDA lifetime gates; no model weights or attention kernels.

Run with ``python -m unittest discover -s tests -v``. These tests skip cleanly
when CUDA is unavailable. Each deliberate asynchronous delay is only milliseconds.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.engine.block_manager import BlockManager
from nanovllm.speculative.jetspec import state as state_module
from nanovllm.speculative.jetspec.state import PagedTargetState


def delay_cuda() -> None:
    # Make missing dependencies reproducible, rather than relying on an idle GPU
    # having completed a preceding stream before a following stream starts.
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(100_000_000)


def create_state(prompt_len: int = 256) -> tuple[PagedTargetState, BlockManager]:
    manager = BlockManager(8, 256)
    pool = torch.zeros((2, 3, 8, 256, 2, 4), dtype=torch.bfloat16, device="cuda")
    prompt_kv = [
        (
            torch.full((prompt_len, 2, 4), float(layer + 1), dtype=pool.dtype, device="cuda"),
            torch.full((prompt_len, 2, 4), float(layer + 11), dtype=pool.dtype, device="cuda"),
        )
        for layer in range(3)
    ]
    hidden = torch.zeros((1, prompt_len, 6), device="cuda")
    tokens = torch.zeros((1, prompt_len + 1), dtype=torch.long, device="cuda")
    slot_builder = state_module._slots_for_blocks

    def delayed_slots(*args, **kwargs):
        # CPU -> CUDA table construction can synchronize this stream. Insert the
        # delay afterwards so the following KV scatter is demonstrably in flight.
        slots = slot_builder(*args, **kwargs)
        delay_cuda()
        return slots

    with patch.object(state_module, "_slots_for_blocks", side_effect=delayed_slots):
        state = PagedTargetState.from_prefill(tokens, prompt_kv, hidden, pool, manager, 256)
    return state, manager


def read_slots(state: PagedTargetState, slots: torch.Tensor) -> torch.Tensor:
    return state.kv_pool[:, :, slots // state.block_size, slots % state.block_size]


def write_slots(state: PagedTargetState, slots: torch.Tensor, values: torch.Tensor) -> None:
    state.kv_pool[:, :, slots // state.block_size, slots % state.block_size] = values


def tree_payload(state: PagedTargetState, nodes: torch.Tensor) -> torch.Tensor:
    shape = (2, 3, nodes.numel(), 2, 4)
    # Distinct node/layer/KV values; integers in BF16's exact small range.
    return torch.arange(2 * 3 * nodes.numel(), device="cuda").reshape(2, 3, -1, 1, 1).remainder(127).to(state.kv_pool.dtype).expand(shape).clone()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class JetSpecKVStreamTest(unittest.TestCase):
    def tearDown(self) -> None:
        torch.cuda.synchronize()

    def assert_released(self, manager: BlockManager) -> None:
        self.assertFalse(manager.used_block_ids)
        self.assertEqual(len(manager.free_block_ids), len(manager.blocks))
        self.assertTrue(all(block.ref_count == 0 for block in manager.blocks))

    def assert_in_flight(self, event: torch.cuda.Event) -> None:
        if hasattr(torch.cuda, "_sleep"):
            self.assertFalse(event.query(), "test handoff did not exercise outstanding GPU work")

    def test_prefill_dependency_first_acquire_on_another_stream(self) -> None:
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            state, manager = create_state()
            self.assert_in_flight(state.scratch._retired)
        try:
            with torch.cuda.stream(consumer):
                state.assert_round_invariant()
                state.reserve_tree(63, max_path_length=4)
                prefix = read_slots(state, state.logical_slots)
                expected = torch.tensor([[1, 2, 3], [11, 12, 13]], device="cuda", dtype=prefix.dtype)
                expected = expected[:, :, None, None, None].expand_as(prefix)
                self.assertTrue(torch.equal(prefix, expected))
                state.abort_tree()
        finally:
            with torch.cuda.stream(consumer):
                state.clear()
        self.assert_released(manager)

    def test_prefill_only_clear_on_another_stream_fences_original_writes(self) -> None:
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            state, manager = create_state()
            completion = state.scratch._retired
            self.assert_in_flight(completion)
        with torch.cuda.stream(consumer):
            state.clear()
        self.assertTrue(completion.query(), "prefill pages were freed before producer completed")
        self.assert_released(manager)
        self.assertEqual(state.clear(), 0)

    def test_commit_stream_mismatch_is_refused_and_clear_still_fences(self) -> None:
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            state, manager = create_state()
            nodes, _ = state.reserve_tree(63, max_path_length=4)
            payload = tree_payload(state, nodes)
            hidden = torch.zeros((1, 63, 6), device="cuda")
            path = torch.tensor([0, 1, 8, 15], device="cuda")
            tokens = torch.cat((state.committed, torch.zeros((1, 4), dtype=torch.long, device="cuda")), dim=1)
            delay_cuda()
            write_slots(state, nodes, payload)
            completion = torch.cuda.Event()
            completion.record(producer)
            self.assert_in_flight(completion)
        try:
            with torch.cuda.stream(consumer):
                with self.assertRaisesRegex(RuntimeError, "lease stream"):
                    state.commit_tree_path(nodes, hidden, path, committed_tokens=tokens)
                state.clear()
                self.assertTrue(completion.query())
        finally:
            with torch.cuda.stream(consumer):
                state.clear()
        self.assert_released(manager)

    def test_retired_accepted_copy_survives_poison_reuse_on_next_stream(self) -> None:
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            state, manager = create_state()
            nodes, _ = state.reserve_tree(63, max_path_length=4)
            payload = tree_payload(state, nodes)
            write_slots(state, nodes, payload)
            path = torch.tensor([0, 7, 32, 62], device="cuda")
            expected = payload.index_select(2, path).clone()
            hidden = torch.zeros((1, 63, 6), device="cuda")
            tokens = torch.cat((state.committed, torch.zeros((1, 4), dtype=torch.long, device="cuda")), dim=1)
            original = state_module.copy_accepted_kv

            def delayed_copy(*args):
                # Validation .tolist() and destination table construction may
                # synchronize preceding work, so delay the actual physical copy.
                delay_cuda()
                return original(*args)

            with patch.object(state_module, "copy_accepted_kv", side_effect=delayed_copy):
                state.commit_tree_path(nodes, hidden, path, committed_tokens=tokens)
            self.assert_in_flight(state.scratch._retired)
            destination = state.logical_slots[-4:]
        try:
            with torch.cuda.stream(consumer):
                reused, _ = state.reserve_tree(63, max_path_length=4)
                poison = torch.full((2, 3, 63, 2, 4), float("nan"), dtype=state.kv_pool.dtype, device="cuda")
                write_slots(state, reused, poison)
                self.assertTrue(torch.equal(read_slots(state, destination), expected))
                self.assertTrue(torch.equal(reused, nodes))
                state.abort_tree()
        finally:
            with torch.cuda.stream(consumer):
                state.clear()
        self.assert_released(manager)

    def test_abort_partial_copy_on_another_stream_fences_destination_release(self) -> None:
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            state, manager = create_state()
            nodes, _ = state.reserve_tree(63, max_path_length=4)
            write_slots(state, nodes, tree_payload(state, nodes))
            prefix = read_slots(state, state.logical_slots).clone()
            hidden = torch.zeros((1, 63, 6), device="cuda")
            path = torch.tensor([0, 1, 8, 15], device="cuda")
            tokens = torch.cat((state.committed, torch.zeros((1, 4), dtype=torch.long, device="cuda")), dim=1)
            original = state_module.copy_accepted_kv

            def fail_after_copy(*args):
                delay_cuda()
                original(*args)
                raise RuntimeError("injected asynchronous partial copy")

            with patch.object(state_module, "copy_accepted_kv", side_effect=fail_after_copy):
                with self.assertRaisesRegex(RuntimeError, "partial copy"):
                    state.commit_tree_path(nodes, hidden, path, committed_tokens=tokens)
            completion = torch.cuda.Event()
            completion.record(producer)
            self.assert_in_flight(completion)
        try:
            with torch.cuda.stream(consumer):
                self.assertEqual(state.abort_tree(), 1)
                self.assertTrue(state.scratch._retired.query())
                self.assertEqual(state.cache_len, 256)
                self.assertTrue(torch.equal(read_slots(state, state.logical_slots), prefix))
                self.assertFalse(state.pending_blocks)
                state.assert_round_invariant()
        finally:
            with torch.cuda.stream(consumer):
                state.clear()
        self.assert_released(manager)

    def test_abort_verify_without_destination_waits_before_scratch_reuse(self) -> None:
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            state, manager = create_state(prompt_len=64)
            nodes, _ = state.reserve_tree(63, max_path_length=4)
            self.assertFalse(state.pending_blocks)
            old_payload = torch.ones((2, 3, 63, 2, 4), dtype=state.kv_pool.dtype, device="cuda")
            delay_cuda()
            write_slots(state, nodes, old_payload)
            completion = torch.cuda.Event()
            completion.record(producer)
            self.assert_in_flight(completion)
        try:
            with torch.cuda.stream(consumer):
                self.assertEqual(state.abort_tree(), 0)
                reused, _ = state.reserve_tree(63, max_path_length=4)
                new_payload = torch.full((2, 3, 63, 2, 4), 3.0, dtype=state.kv_pool.dtype, device="cuda")
                write_slots(state, reused, new_payload)
                self.assertTrue(torch.equal(read_slots(state, reused), new_payload))
                state.abort_tree()
        finally:
            with torch.cuda.stream(consumer):
                state.clear()
        self.assert_released(manager)


if __name__ == "__main__":
    unittest.main()
