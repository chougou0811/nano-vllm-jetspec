"""Model-free request lifecycle gates for the packed runtime step boundary."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanovllm.engine.block_manager import BlockManager
from nanovllm.speculative.jetspec.batch_runtime import JetSpecBatchRuntime, JetSpecRequest
from nanovllm.speculative.jetspec.state import BatchTreeTransaction, PagedTargetState, TreeScratchArena


class FakeEvent:
    def __init__(self, **kwargs):
        pass

    def record(self):
        pass

    def elapsed_time(self, other):
        return 1.0


def make_runtime():
    runner = JetSpecBatchRuntime.__new__(JetSpecBatchRuntime)
    runner.kv_pool = torch.zeros(2, 3, 16, 256, 2, 4)
    runner.block_manager = BlockManager(16, 256)
    runner.block_size, runner.tree_depth, runner.tree_width = 256, 15, 7
    runner.max_verify_tokens, runner.max_model_len = 4096, 4096
    runner.arena = TreeScratchArena(runner.kv_pool, runner.block_manager, 256)
    runner.requests, runner._active_transaction = {}, None
    runner._closed, runner._reference_mode = False, False
    runner.eos_token_ids = {63}
    runner.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids))

    def build(root, logits, depth, width, budget, device):
        n = min(budget, 3)
        return SimpleNamespace(token_ids=torch.arange(root, root + n),
                               depth=torch.arange(n), num_nodes=n,
                               parent_indices=torch.arange(-1, n - 1),
                               ancestor=torch.ones(n, n, dtype=torch.bool).tril())

    runner.tree_algorithm = SimpleNamespace(build=build)

    def verify(requests, trees, transaction, metadata):
        tokens = torch.cat([tree.token_ids for tree in trees])
        logits = torch.zeros(tokens.numel(), 64)
        logits[torch.arange(tokens.numel()), tokens + 1] = 1
        slots = transaction.packed_node_slots
        runner.kv_pool[:, :, slots // 256, slots % 256] = 17
        return logits, torch.ones(tokens.numel(), 6)

    runner._verify_batch = verify
    return runner


def add_request(runner, request_id, limit=4, prompt_len=4):
    kv = [(torch.zeros(prompt_len, 2, 4), torch.ones(prompt_len, 2, 4)) for _ in range(3)]
    tokens = torch.zeros(1, prompt_len + 1, dtype=torch.long)
    tokens[0, -1] = 10
    state = PagedTargetState.from_prefill(tokens, kv, torch.zeros(1, prompt_len, 6),
                                         runner.kv_pool, runner.block_manager, 256)
    drafter = Mock()
    drafter.propose_logits.return_value = torch.zeros(1, 15, 64)
    request = JetSpecRequest(request_id, state, drafter, 3, limit, False,
                            prompt_len, [10], time.perf_counter())
    runner.requests[request_id] = request
    return request


class JetSpecBatchRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.event_patch = patch("torch.cuda.Event", FakeEvent)
        self.event_patch.start()
        self.runner = make_runtime()

    def tearDown(self):
        self.runner.close()
        self.assertFalse(self.runner.block_manager.used_block_ids)
        self.event_patch.stop()

    def test_independent_limits_finish_and_shared_arena_survival(self):
        short, long = add_request(self.runner, "short", 2), add_request(self.runner, "long", 4)
        record = self.runner.step([long, short])
        self.assertEqual(record["request_ids"], ["long", "short"])
        self.assertEqual(short.output_ids, [10, 11])
        self.assertEqual(long.output_ids, [10, 11, 12, 13])
        self.assertEqual(short.state.cache_len, short.prompt_length + 1)
        self.assertEqual(long.state.cache_len, long.prompt_length + 3)
        arena_pages = list(self.runner.arena.blocks)
        for request in (short, long):
            result = self.runner.finish(request)
            self.assertEqual(result["state_invariant"]["kv_length"],
                             result["state_invariant"]["committed_minus_one"])
            self.assertIs(self.runner.finish(request), result)
        self.assertEqual(self.runner.arena.blocks, arena_pages)
        self.assertEqual(self.runner.block_manager.used_block_ids, set(arena_pages))
        new = add_request(self.runner, "new", 4)
        self.runner.step([new])
        self.assertEqual(self.runner.arena.blocks, arena_pages)
        self.runner.finish(new)

    def test_duplicate_foreign_and_query_budget_are_rejected(self):
        request = add_request(self.runner, "a")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.runner.step([request, request])
        other = make_runtime()
        foreign = add_request(other, "b")
        try:
            with self.assertRaisesRegex(ValueError, "belong"):
                self.runner.step([foreign])
        finally:
            other.close()
        self.runner.max_verify_tokens = 2
        with self.assertRaisesRegex(ValueError, "token budget"):
            self.runner.step([request])
        self.assertIsNone(self.runner._active_transaction)
        request.state.assert_round_invariant()

    def test_verify_failure_rolls_back_all_and_retry_succeeds(self):
        requests = [add_request(self.runner, "a"), add_request(self.runner, "b")]
        original = self.runner._verify_batch

        def fail_after_verify(*args):
            original(*args)
            raise RuntimeError("injected verify error")

        self.runner._verify_batch = fail_after_verify
        with self.assertRaisesRegex(RuntimeError, "verify error"):
            self.runner.step(requests)
        for request in requests:
            self.assertEqual(request.output_ids, [10])
            self.assertEqual(request.state.cache_len, request.prompt_length)
            self.assertFalse(request.state.pending_blocks)
            request.state.assert_round_invariant()
        self.assertIsNone(self.runner._active_transaction)
        self.assertFalse(self.runner.arena.active)
        self.runner._verify_batch = original
        self.runner.step(requests)
        for request in requests:
            self.runner.finish(request)

    def test_postcommit_reporting_failure_keeps_all_output_states_consistent(self):
        requests = [add_request(self.runner, "a"), add_request(self.runner, "b")]
        original = BatchTreeTransaction.commit

        def report_failure(transaction, *args):
            original(transaction, *args)
            raise RuntimeError("injected postcommit reporting error")

        with patch.object(BatchTreeTransaction, "commit", report_failure):
            with self.assertRaisesRegex(RuntimeError, "postcommit reporting"):
                self.runner.step(requests)
        for request in requests:
            self.assertEqual(request.output_ids, [10, 11, 12, 13])
            self.assertEqual(request.state.committed.shape[1], request.prompt_length + len(request.output_ids))
            self.assertEqual(len(request.rounds), 1)
            self.assertTrue(request.finished)
            request.state.assert_round_invariant()
            self.runner.finish(request)
        self.assertIsNone(self.runner._active_transaction)

    def test_eos_is_uncached_and_cancel_does_not_free_shared_scratch(self):
        request = add_request(self.runner, "eos", 16)
        self.runner.eos_token_ids = {11}
        self.runner.step([request])
        self.assertEqual(request.output_ids, [10, 11])
        self.assertEqual(request.state.cache_len, request.prompt_length + 1)
        self.runner.finish(request)
        request = add_request(self.runner, "cancel", 16)
        arena_pages = list(self.runner.arena.blocks)
        result = self.runner.cancel(request)
        self.assertTrue(result["cancelled"])
        self.assertEqual(self.runner.arena.blocks, arena_pages)
        self.assertEqual(self.runner.block_manager.used_block_ids, set(arena_pages))


if __name__ == "__main__":
    unittest.main()
