"""Model-free continuous-serving lifecycle tests using the real paged KV state."""
from itertools import count
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.speculative.jetspec.state import TreeScratchArena
from tests.test_jetspec_batch_runtime import FakeEvent, make_runtime


def make_serving_runtime(num_blocks=16, *, max_verify_tokens=4096):
    """Actual runtime/transactions/allocator with cheap deterministic Target ops.

    Token t always predicts t+1, independently of execution shape. This removes
    numerical-drift distractions while exercising ownership, truncation,
    preemption, streaming and scheduler integration without CUDA/model weights.
    """
    runtime = make_runtime()
    runtime.kv_pool = torch.zeros(2, 3, num_blocks, 256, 2, 4)
    runtime.block_manager = BlockManager(num_blocks, 256)
    runtime.arena = TreeScratchArena(runtime.kv_pool, runtime.block_manager, 256)
    runtime.max_tree_budget = 63
    runtime.max_verify_tokens = max_verify_tokens
    runtime._ids = count()
    runtime.target_layer_ids = (0, 1, 2)
    runtime.tokenizer = SimpleNamespace(
        encode=lambda text: [int(token) for token in text.split()],
        decode=lambda ids, **kwargs: " ".join(map(str, ids)),
        eos_token_id=63,
    )

    def dense(ids, positions, past, mask, target_layer_ids):
        hidden = torch.zeros(ids.numel(), 6)
        hidden[:, 0] = ids
        values = ids.float()[:, None, None].expand(-1, 2, 4).clone()
        kv = [(values + layer, values + layer + 1) for layer in range(3)]
        return hidden, kv, hidden.clone()

    def head(hidden):
        logits = torch.zeros(hidden.shape[0], 64)
        next_ids = hidden[:, 0].long() + 1
        logits[torch.arange(hidden.shape[0]), next_ids] = 1
        return logits

    runtime.target = SimpleNamespace(model=SimpleNamespace(forward_dense=dense), lm_head=head)

    def drafter():
        result = Mock()
        result.propose_logits.return_value = torch.zeros(1, 15, 64)
        return result

    runtime._new_drafter = drafter
    return runtime


def make_engine(runtime, *, max_num_seqs=2):
    """No heavyweight LLM constructor; production engine methods remain intact."""
    engine = LLMEngine.__new__(LLMEngine)
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=runtime.max_verify_tokens,
        max_model_len=runtime.max_model_len,
        eos=63, num_kvcache_blocks=len(runtime.block_manager.blocks),
        kvcache_block_size=256,
    )
    engine.model_runner = SimpleNamespace(
        model=runtime.target, kv_cache=runtime.kv_pool,
        config=config, block_size=256, world_size=1, enforce_eager=True,
        call=Mock(),
    )
    engine.tokenizer = runtime.tokenizer
    engine.scheduler = Scheduler(config)
    engine.scheduler.block_manager = runtime.block_manager
    engine._jetspec_batch_runtime = (("fake-draft", 15, 7, 63), runtime)
    engine.ps, engine.events = [], []
    return engine


class ContinuousRuntimeFixture(unittest.TestCase):
    def setUp(self):
        self.event_patch = patch("torch.cuda.Event", FakeEvent)
        self.event_patch.start()
        self.runtime = make_serving_runtime()
        self.serving = None

    def tearDown(self):
        if self.serving is not None:
            self.serving.close()
        self.runtime.close()
        self.assertFalse(self.runtime.block_manager.used_block_ids)
        self.event_patch.stop()

    def scheduler(self, **kwargs):
        from nanovllm.engine.jetspec_scheduler import JetSpecScheduler
        self.serving = JetSpecScheduler(self.runtime, **kwargs)
        return self.serving

    def replace_runtime(self, num_blocks, **kwargs):
        self.runtime.close()
        self.runtime = make_serving_runtime(num_blocks, **kwargs)
        return self.runtime

    def drain(self, scheduler, *, max_steps=64):
        steps = []
        while not scheduler.is_finished():
            self.assertLess(len(steps), max_steps, "serving made no bounded progress")
            steps.append(scheduler.step())
        return steps


class JetSpecContinuousRuntimeTest(ContinuousRuntimeFixture):
    def test_suspend_resume_preserves_uncached_anchor_and_output_history(self):
        request = self.runtime.create_request([1, 2, 3], request_id="a", max_new_tokens=12, tree_budget=3)
        self.runtime.step([request])
        output = list(request.output_ids)
        committed = request.state.committed.clone()
        arena_pages = list(self.runtime.arena.blocks)
        snapshot = self.runtime.suspend(request)
        self.assertNotIn("a", self.runtime.requests)
        self.assertEqual(self.runtime.arena.blocks, arena_pages)
        self.assertEqual(self.runtime.block_manager.used_block_ids, set(arena_pages))
        resumed = self.runtime.resume(snapshot)
        self.assertEqual(resumed.request_id, "a")
        self.assertEqual(resumed.output_ids, output)
        self.assertTrue(torch.equal(resumed.state.committed, committed))
        self.assertEqual(resumed.state.cache_len, resumed.prompt_length + len(output) - 1)
        resumed.state.assert_round_invariant()
        self.runtime.step([resumed])
        self.assertEqual(resumed.output_ids[:len(output)], output)
        self.assertEqual(resumed.output_ids, list(range(4, 4 + len(resumed.output_ids))))

    def test_effective_round_budgets_do_not_mutate_request_caps(self):
        requests = [self.runtime.create_request([3], request_id=name,
                    max_new_tokens=10, tree_budget=3) for name in ("a", "b")]
        plan = self.runtime.estimate_step_capacity(requests, [1, 2])
        self.assertTrue(plan["feasible"])
        self.assertEqual(plan["total_query_tokens"], 3)
        report = self.runtime.step(requests, tree_budgets=[1, 2])
        self.assertEqual(report["total_query_tokens"], 3)
        self.assertEqual([r.tree_budget for r in requests], [3, 3])
        self.assertEqual([len(r.output_ids) for r in requests], [2, 3])
        for request in requests:
            request.state.assert_round_invariant()
        with self.assertRaises(ValueError):
            self.runtime.step(requests, tree_budgets=[4, 1])

    def test_output_cap_bounds_destination_reserve_at_page_boundary(self):
        self.replace_runtime(2)
        request = self.runtime.create_request([3] * 255, request_id="a",
                                               max_new_tokens=2, tree_budget=3)
        estimate = self.runtime.estimate_step_capacity([request])
        self.assertTrue(estimate["feasible"])
        self.assertEqual(estimate["destination_blocks"], 0)
        report = self.runtime.step([request])
        self.assertTrue(request.finished)
        self.assertEqual(request.state.cache_len, 256)
        self.assertEqual(len(request.state.owned_blocks), 1)
        self.assertEqual(len(self.runtime.arena.blocks), 1)
        self.assertEqual(report["lifecycle"]["kv_copy_bytes"],
                         self.runtime.kv_pool[:, :, 0, 0].numel() * self.runtime.kv_pool.element_size())

    def test_failed_resume_retains_snapshot_and_leaks_no_canonical_pages(self):
        request = self.runtime.create_request([3], request_id="a", max_new_tokens=10, tree_budget=3)
        self.runtime.step([request])
        snapshot = self.runtime.suspend(request)
        output, committed = list(snapshot["output_ids"]), list(snapshot["committed_tokens"])
        before = set(self.runtime.block_manager.used_block_ids)
        original = self.runtime.target.model.forward_dense
        self.runtime.target.model.forward_dense = Mock(side_effect=RuntimeError("resume prefill failed"))
        with self.assertRaisesRegex(RuntimeError, "resume prefill failed"):
            self.runtime.resume(snapshot)
        self.assertEqual(self.runtime.block_manager.used_block_ids, before)
        self.assertFalse(self.runtime.requests)
        self.assertEqual(snapshot["output_ids"], output)
        self.assertEqual(snapshot["committed_tokens"], committed)
        self.runtime.target.model.forward_dense = original
        resumed = self.runtime.resume(snapshot)
        self.assertEqual(resumed.output_ids, output)
        resumed.state.assert_round_invariant()

    def test_finish_draft_reset_failure_keeps_published_result_and_detaches_released_state(self):
        request = self.runtime.create_request([3], request_id="a", max_new_tokens=4, tree_budget=3)
        self.runtime.step([request])
        self.assertTrue(request.finished)
        output = list(request.output_ids)
        request.drafter.reset_cache.side_effect = RuntimeError("finish Draft reset failed")
        with self.assertRaisesRegex(RuntimeError, "finish Draft reset"):
            self.runtime.finish(request)
        self.assertNotIn("a", self.runtime.requests)
        self.assertEqual(request.result["token_ids"], output)
        self.assertFalse(request.state.owned_blocks)
        self.assertEqual(self.runtime.block_manager.used_block_ids, set(self.runtime.arena.blocks))
        self.assertIs(self.runtime.finish(request), request.result)
        request.drafter.reset_cache.assert_called_once()

    def test_suspend_draft_reset_failure_preserves_registry_canonical_bytes_and_progress(self):
        request = self.runtime.create_request([3], request_id="a", max_new_tokens=12, tree_budget=3)
        self.runtime.step([request])
        slots = request.state.logical_slots
        before_kv = self.runtime.kv_pool[:, :, slots // 256, slots % 256].clone()
        before_tokens = request.state.committed.clone()
        before_output = list(request.output_ids)
        before_used = set(self.runtime.block_manager.used_block_ids)
        before_owned = list(request.state.owned_blocks)
        request.drafter.reset_cache.side_effect = RuntimeError("suspend Draft reset failed")
        try:
            with self.assertRaisesRegex(RuntimeError, "suspend Draft reset"):
                self.runtime.suspend(request)
            self.assertIs(self.runtime.requests["a"], request)
            self.assertEqual(self.runtime.block_manager.used_block_ids, before_used)
            self.assertEqual(request.state.owned_blocks, before_owned)
            self.assertEqual(request.output_ids, before_output)
            self.assertTrue(torch.equal(request.state.committed, before_tokens))
            self.assertTrue(torch.equal(self.runtime.kv_pool[:, :, slots // 256, slots % 256], before_kv))
            request.state.assert_round_invariant()
        finally:
            request.drafter.reset_cache.side_effect = None


class JetSpecContinuousSchedulerTest(ContinuousRuntimeFixture):
    def test_dynamic_arrival_short_finish_and_ragged_next_step(self):
        scheduler = self.scheduler(max_num_seqs=2, max_admissions_per_step=1)
        scheduler.add_request([1, 2, 3], request_id="a", max_new_tokens=16, tree_budget=3)
        first = scheduler.step()
        self.assertIn("a", first["admitted_ids"])
        a = self.runtime.requests["a"]
        old_output = list(a.output_ids)
        old_prefix = a.state.committed.clone()
        scratch = list(self.runtime.arena.blocks)
        scheduler.add_request([5], request_id="short", max_new_tokens=1, tree_budget=1)
        scheduler.add_request([5], request_id="ragged", max_new_tokens=7, tree_budget=2)
        self.assertTrue(torch.equal(a.state.committed, old_prefix))
        second = scheduler.step()
        self.assertIn("short", second["admitted_ids"])
        self.assertTrue(any(e["kind"] == "finished" and e["request_id"] == "short"
                            for e in second["events"]))
        self.assertNotIn("short", self.runtime.requests)
        self.assertEqual(a.output_ids[:len(old_output)], old_output)
        third = scheduler.step()
        self.assertIn("ragged", third["admitted_ids"])
        self.assertEqual(third["verification"]["request_ids"], ["a", "ragged"])
        self.assertEqual(third["verification"]["total_query_tokens"], 5)
        self.assertEqual(self.runtime.arena.blocks, scratch)
        for request in self.runtime.requests.values():
            request.state.assert_round_invariant()
        self.drain(scheduler)

    def test_stream_deltas_are_exactly_once_and_terminal_output_is_full(self):
        scheduler = self.scheduler(max_num_seqs=2)
        scheduler.add_request([3], request_id="a", max_new_tokens=11, tree_budget=3)
        scheduler.add_request([7], request_id="b", max_new_tokens=6, tree_budget=2)
        emitted, terminal = {"a": [], "b": []}, {}
        for step in self.drain(scheduler):
            for event in step["events"]:
                if event["kind"] == "tokens":
                    emitted[event["request_id"]].extend(event["token_ids"])
                    self.assertEqual(len(emitted[event["request_id"]]), event["output_length"])
                elif event["kind"] == "finished":
                    self.assertNotIn(event["request_id"], terminal)
                    terminal[event["request_id"]] = event["token_ids"]
        self.assertEqual(emitted, terminal)
        self.assertEqual(terminal["a"], list(range(4, 15)))
        self.assertEqual(terminal["b"], list(range(8, 14)))
        self.assertFalse(self.runtime.requests)
        self.assertEqual(self.runtime.block_manager.used_block_ids, set(self.runtime.arena.blocks))

    def test_waiting_and_running_cancel_release_only_request_pages(self):
        scheduler = self.scheduler(max_num_seqs=1)
        scheduler.add_request([3], request_id="running", max_new_tokens=16, tree_budget=3)
        scheduler.add_request([7], request_id="waiting", max_new_tokens=6, tree_budget=2)
        scheduler.step()
        self.assertIn("running", self.runtime.requests)
        self.assertNotIn("waiting", self.runtime.requests)
        scratch = list(self.runtime.arena.blocks)
        waiting_result = scheduler.cancel("waiting")
        running_result = scheduler.cancel("running")
        self.assertEqual(waiting_result["kind"], "cancelled")
        self.assertEqual(waiting_result["token_ids"], [])
        self.assertEqual(running_result["kind"], "cancelled")
        self.assertTrue(running_result["token_ids"])
        self.assertEqual(self.runtime.arena.blocks, scratch)
        self.assertEqual(self.runtime.block_manager.used_block_ids, set(scratch))
        events = scheduler.step()["events"]
        self.assertEqual({e["request_id"] for e in events if e["kind"] == "cancelled"},
                         {"waiting", "running"})
        self.assertIsNone(scheduler.cancel("unknown"))
        self.assertTrue(scheduler.is_finished())

    def test_query_budget_and_rotation_give_every_request_progress(self):
        self.replace_runtime(16, max_verify_tokens=3)
        scheduler = self.scheduler(max_num_seqs=3, max_verify_tokens=3)
        for name in ("a", "b", "c"):
            scheduler.add_request([3], request_id=name, max_new_tokens=20, tree_budget=3)
        seen = set()
        for _ in range(5):
            report = scheduler.step()
            verify = report["verification"]
            if verify:
                self.assertLessEqual(verify["total_query_tokens"], 3)
                seen.update(verify["request_ids"])
        self.assertEqual(seen, {"a", "b", "c"})
        self.assertLessEqual(len(self.runtime.requests), 3)
        self.drain(scheduler)

    def test_prefill_budget_preserves_fifo_and_leaves_decode_progress(self):
        scheduler = self.scheduler(max_num_seqs=4, max_admissions_per_step=2,
                                   max_prefill_tokens=8)
        for name, size in (("a", 5), ("b", 5), ("c", 2)):
            scheduler.add_request([3] * size, request_id=name,
                                  max_new_tokens=12, tree_budget=3)
        first = scheduler.step()
        self.assertEqual(first["admitted_ids"], ["a"])
        self.assertIn("a", first["verification"]["request_ids"])
        previous = list(self.runtime.requests["a"].output_ids)
        second = scheduler.step()
        self.assertEqual(second["admitted_ids"], ["b", "c"])
        self.assertGreater(len(self.runtime.requests["a"].output_ids), len(previous))
        self.assertEqual(self.runtime.requests["a"].output_ids[:len(previous)], previous)
        self.drain(scheduler)

    def test_duplicate_ids_and_invalid_budgets_are_rejected_before_admission(self):
        scheduler = self.scheduler(max_num_seqs=2, max_verify_tokens=3)
        scheduler.add_request([3], request_id="a", tree_budget=3)
        with self.assertRaises(ValueError):
            scheduler.add_request([3], request_id="a", tree_budget=3)
        with self.assertRaises(ValueError):
            scheduler.add_request([], request_id="empty", tree_budget=3)
        with self.assertRaises(ValueError):
            scheduler.add_request([3], request_id="negative", max_new_tokens=-1, tree_budget=3)
        with self.assertRaises(ValueError):
            scheduler.add_request([3], request_id="large", tree_budget=64)
        self.assertFalse(self.runtime.requests)
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_zero_token_request_has_terminal_delivery_without_prefill(self):
        scheduler = self.scheduler(max_num_seqs=1)
        scheduler.add_request([3], request_id="zero", max_new_tokens=0, tree_budget=3)
        self.assertFalse(scheduler.is_finished())
        self.assertFalse(self.runtime.requests)
        self.assertFalse(self.runtime.block_manager.used_block_ids)
        with self.assertRaises(ValueError):
            scheduler.add_request([3], request_id="zero", max_new_tokens=0, tree_budget=3)
        report = scheduler.step()
        self.assertIsNone(report["verification"])
        self.assertEqual(report["events"][0]["kind"], "finished")
        self.assertEqual(report["events"][0]["token_ids"], [])
        self.assertTrue(scheduler.is_finished())
        # Once terminal delivery is consumed, reusing a caller-supplied ID is safe.
        scheduler.add_request([3], request_id="zero", max_new_tokens=0, tree_budget=3)
        self.assertEqual(len(scheduler.drain_events()), 1)
        self.assertTrue(scheduler.is_finished())

    def test_request_cap_can_exceed_query_budget_without_mutating_it(self):
        self.replace_runtime(16, max_verify_tokens=3)
        scheduler = self.scheduler(max_num_seqs=1, max_verify_tokens=3)
        scheduler.add_request([3], request_id="a", max_new_tokens=10, tree_budget=4)
        report = scheduler.step()
        self.assertLessEqual(report["verification"]["total_query_tokens"], 3)
        self.assertEqual(self.runtime.requests["a"].tree_budget, 4)
        self.drain(scheduler)

    def test_fair_waterfill_gives_three_residents_one_node_each_under_q3(self):
        self.replace_runtime(16, max_verify_tokens=3)
        scheduler = self.scheduler(max_num_seqs=3, max_verify_tokens=3,
                                   max_admissions_per_step=3)
        for name in ("a", "b", "c"):
            scheduler.add_request([3], request_id=name, max_new_tokens=6, tree_budget=3)
        report = scheduler.step()
        self.assertEqual(report["verification"]["request_ids"], ["a", "b", "c"])
        self.assertEqual(report["verification"]["total_query_tokens"], 3)
        for request in self.runtime.requests.values():
            self.assertEqual(request.tree_budget, 3)
            self.assertEqual(request.rounds[-1]["effective_tree_budget"], 1)
            self.assertEqual(request.output_ids, [4, 5])
            request.drafter.propose_logits.assert_not_called()
        self.drain(scheduler)

    def test_kv_pressure_preempts_recomputes_and_never_reemits_history(self):
        self.replace_runtime(3)
        scheduler = self.scheduler(max_num_seqs=2, max_admissions_per_step=1)
        scheduler.add_request([3] * 250, request_id="a", max_new_tokens=13, tree_budget=3)
        scheduler.add_request([7] * 250, request_id="b", max_new_tokens=10, tree_budget=3)
        emitted, terminal, preempted, resumed = {"a": [], "b": []}, {}, set(), set()
        for report in self.drain(scheduler, max_steps=40):
            self.assertLessEqual(report["capacity"]["allocator_used_blocks"], 3)
            preempted.update(report["preempted_ids"])
            resumed.update(report["resumed_ids"])
            for event in report["events"]:
                if event["kind"] == "tokens":
                    emitted[event["request_id"]].extend(event["token_ids"])
                    self.assertEqual(len(emitted[event["request_id"]]), event["output_length"])
                elif event["kind"] == "finished":
                    terminal[event["request_id"]] = event["token_ids"]
                elif event["kind"] == "error":
                    self.fail(event["reason"])
        self.assertTrue(preempted, "test must actually exercise recompute preemption")
        self.assertTrue(preempted <= resumed)
        self.assertEqual(emitted, terminal)
        self.assertEqual(terminal["a"], list(range(4, 17)))
        self.assertEqual(terminal["b"], list(range(8, 18)))

    def test_cancel_suspended_request_does_not_recreate_or_emit_anchor_again(self):
        self.replace_runtime(3)
        scheduler = self.scheduler(max_num_seqs=2, max_admissions_per_step=1)
        for name, token in (("a", 3), ("b", 7)):
            scheduler.add_request([token] * 250, request_id=name,
                                  max_new_tokens=16, tree_budget=3)
        victim = None
        for _ in range(16):
            report = scheduler.step()
            if report["preempted_ids"]:
                victim = report["preempted_ids"][0]
                break
        self.assertIsNotNone(victim)
        self.assertNotIn(victim, self.runtime.requests)
        saved_output = list(scheduler.requests[victim].output_ids)
        before = set(self.runtime.block_manager.used_block_ids)
        confirmation = scheduler.cancel(victim)
        self.assertEqual(confirmation["token_ids"], saved_output)
        self.assertEqual(self.runtime.block_manager.used_block_ids, before)
        later = self.drain(scheduler)
        self.assertFalse(any(victim in report["resumed_ids"] for report in later))
        self.assertFalse(any(event["kind"] == "tokens" and event["request_id"] == victim
                             for report in later for event in report["events"]))

    def test_precommit_verify_error_retains_delivery_and_other_waiting_request(self):
        scheduler = self.scheduler(max_num_seqs=1)
        scheduler.add_request([3], request_id="failed", max_new_tokens=10, tree_budget=3)
        scheduler.add_request([7], request_id="waiting", max_new_tokens=4, tree_budget=3)
        original = self.runtime._verify_batch

        def fail(*args):
            original(*args)
            raise RuntimeError("verify injected before commit")

        self.runtime._verify_batch = fail
        with self.assertRaisesRegex(RuntimeError, "before commit"):
            scheduler.step()
        self.assertFalse(self.runtime.requests)
        self.assertIn("waiting", scheduler.requests)
        self.assertEqual(self.runtime.block_manager.used_block_ids, set(self.runtime.arena.blocks))
        events = scheduler.drain_events()
        self.assertEqual([e["token_ids"] for e in events if e["kind"] == "tokens"], [[4]])
        errors = [e for e in events if e["kind"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["token_ids"], [4])
        self.runtime._verify_batch = original
        self.drain(scheduler)

    def test_postcommit_reporting_error_delivers_committed_progress_exactly_once(self):
        scheduler = self.scheduler(max_num_seqs=1)
        scheduler.add_request([3], request_id="a", max_new_tokens=10, tree_budget=3)
        original = self.runtime.step

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("report injected after commit")

        self.runtime.step = fail
        with self.assertRaisesRegex(RuntimeError, "after commit"):
            scheduler.step()
        self.assertFalse(self.runtime.requests)
        self.assertEqual(self.runtime.block_manager.used_block_ids, set(self.runtime.arena.blocks))
        events = scheduler.drain_events()
        emitted = [token for event in events if event["kind"] == "tokens" for token in event["token_ids"]]
        self.assertEqual(emitted, [4, 5, 6, 7])
        errors = [e for e in events if e["kind"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["token_ids"], emitted)
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(scheduler.step()["events"], [])

    def test_unfit_singleton_emits_capacity_error_without_infinite_retry(self):
        self.replace_runtime(1)
        scheduler = self.scheduler(max_num_seqs=1)
        scheduler.add_request([3], request_id="unfit", max_new_tokens=4, tree_budget=3)
        report = scheduler.step()
        self.assertIsNone(report["verification"])
        errors = [e for e in report["events"] if e["kind"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("KV capacity", errors[0]["reason"])
        self.assertTrue(scheduler.is_finished())
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_temporary_external_capacity_pressure_defers_admission_then_recovers(self):
        self.replace_runtime(3)
        scheduler = self.scheduler(max_num_seqs=1)
        held = self.runtime.block_manager.reserve_provisional(2)
        try:
            scheduler.add_request([3], request_id="a", max_new_tokens=4, tree_budget=3)
            report = scheduler.step()
            self.assertTrue(report["blocked"])
            self.assertEqual(report["blocked_reason"], "waiting_for_kv_admission")
            self.assertIsNone(report["verification"])
            self.assertFalse(report["events"])
            self.assertIn("a", scheduler.requests)
            self.assertFalse(self.runtime.requests)
            self.assertEqual(self.runtime.block_manager.used_block_ids, set(held))
        finally:
            self.runtime.block_manager.release_provisional(held)
        reports = self.drain(scheduler)
        self.assertTrue(any("a" in report["admitted_ids"] for report in reports))
        self.assertFalse(any(event["kind"] == "error" for report in reports for event in report["events"]))

    def test_temporary_external_capacity_pressure_keeps_live_prefix_then_recovers(self):
        self.replace_runtime(3)
        scheduler = self.scheduler(max_num_seqs=1)
        scheduler.add_request([3] * 250, request_id="a", max_new_tokens=10, tree_budget=3)
        scheduler.step()
        scheduler.step()
        request = self.runtime.requests["a"]
        self.assertEqual(request.state.cache_len, 256)
        before_tokens = list(request.output_ids)
        before_committed = request.state.committed.clone()
        held = self.runtime.block_manager.reserve_provisional(1)
        try:
            report = scheduler.step()
            self.assertTrue(report["blocked"])
            self.assertEqual(report["blocked_reason"], "waiting_for_kv_verification")
            self.assertIsNone(report["verification"])
            self.assertFalse(report["events"])
            self.assertIs(self.runtime.requests["a"], request)
            self.assertEqual(request.output_ids, before_tokens)
            self.assertTrue(torch.equal(request.state.committed, before_committed))
            request.state.assert_round_invariant()
        finally:
            self.runtime.block_manager.release_provisional(held)
        recovered = scheduler.step()
        self.assertFalse(recovered["blocked"])
        finished = next(e for e in recovered["events"] if e["kind"] == "finished")
        self.assertEqual(finished["token_ids"], list(range(4, 14)))
        self.assertTrue(scheduler.is_finished())

    def test_finish_cleanup_failure_queues_one_error_and_full_output_without_kv_leak(self):
        scheduler = self.scheduler(max_num_seqs=1)
        scheduler.add_request([3], request_id="a", max_new_tokens=7, tree_budget=3)
        first = scheduler.step()
        emitted = [token for event in first["events"] if event["kind"] == "tokens" for token in event["token_ids"]]
        request = self.runtime.requests["a"]
        request.drafter.reset_cache.side_effect = RuntimeError("terminal Draft reset failed")
        with self.assertRaisesRegex(RuntimeError, "terminal Draft reset"):
            scheduler.step()
        self.assertFalse(self.runtime.requests)
        self.assertFalse(scheduler.requests)
        self.assertIsNotNone(request.result)
        self.assertEqual(self.runtime.block_manager.used_block_ids, set(self.runtime.arena.blocks))
        events = scheduler.drain_events()
        emitted += [token for event in events if event["kind"] == "tokens" for token in event["token_ids"]]
        errors = [event for event in events if event["kind"] == "error"]
        self.assertEqual(emitted, list(range(4, 11)))
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["token_ids"], emitted)
        self.assertEqual(errors[0]["result"]["token_ids"], emitted)
        request.drafter.reset_cache.assert_called_once()
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(scheduler.step()["events"], [])

    def test_cancel_cleanup_failure_queues_one_error_and_repeat_cancel_is_idempotent(self):
        scheduler = self.scheduler(max_num_seqs=1)
        scheduler.add_request([3], request_id="a", max_new_tokens=12, tree_budget=3)
        scheduler.step()
        request = self.runtime.requests["a"]
        output = list(request.output_ids)
        request.drafter.reset_cache.side_effect = RuntimeError("cancel Draft reset failed")
        with self.assertRaisesRegex(RuntimeError, "cancel Draft reset"):
            scheduler.cancel("a")
        self.assertFalse(self.runtime.requests)
        self.assertFalse(scheduler.requests)
        self.assertTrue(request.result["cancelled"])
        self.assertIsNone(scheduler.cancel("a"))
        events = scheduler.step()["events"]
        self.assertFalse(any(e["kind"] == "tokens" for e in events))
        errors = [event for event in events if event["kind"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["token_ids"], output)
        request.drafter.reset_cache.assert_called_once()
        self.assertTrue(scheduler.is_finished())

    def test_preempt_cleanup_failure_keeps_resident_and_canonical_bytes_then_retry_succeeds(self):
        self.replace_runtime(3)
        scheduler = self.scheduler(max_num_seqs=2, max_admissions_per_step=1)
        for name, token in (("a", 3), ("b", 7)):
            scheduler.add_request([token] * 250, request_id=name,
                                  max_new_tokens=16, tree_budget=3)
        scheduler.step()
        scheduler.step()
        victim = self.runtime.requests["b"]
        slots = victim.state.logical_slots
        before_kv = self.runtime.kv_pool[:, :, slots // 256, slots % 256].clone()
        before_tokens = victim.state.committed.clone()
        before_output = list(victim.output_ids)
        before_used = set(self.runtime.block_manager.used_block_ids)
        victim.drafter.reset_cache.side_effect = RuntimeError("preemption Draft reset failed")
        try:
            with self.assertRaisesRegex(RuntimeError, "preemption Draft reset"):
                scheduler.step()
            self.assertIs(self.runtime.requests["b"], victim)
            self.assertIs(scheduler.requests["b"].request, victim)
            self.assertIsNone(scheduler.requests["b"].snapshot)
            self.assertEqual(scheduler.requests["b"].status, "running")
            self.assertIn("b", scheduler.running)
            self.assertEqual(self.runtime.block_manager.used_block_ids, before_used)
            self.assertEqual(victim.output_ids, before_output)
            self.assertTrue(torch.equal(victim.state.committed, before_tokens))
            self.assertTrue(torch.equal(self.runtime.kv_pool[:, :, slots // 256, slots % 256], before_kv))
            victim.state.assert_round_invariant()
        finally:
            victim.drafter.reset_cache.side_effect = None
        recovered = scheduler.step()
        self.assertIn("b", recovered["preempted_ids"])
        self.drain(scheduler)

    def test_prefill_eos_and_ignore_eos_remain_request_local(self):
        self.runtime.eos_token_ids = {5}
        scheduler = self.scheduler(max_num_seqs=2)
        scheduler.add_request([4], request_id="eos", max_new_tokens=8, tree_budget=3)
        scheduler.add_request([4], request_id="ignore", max_new_tokens=4,
                              tree_budget=3, ignore_eos=True)
        terminal = {}
        for report in self.drain(scheduler):
            for event in report["events"]:
                if event["kind"] == "finished":
                    terminal[event["request_id"]] = event["token_ids"]
        self.assertEqual(terminal, {"eos": [5], "ignore": [5, 6, 7, 8]})

    def test_close_is_idempotent_and_reclaims_waiting_resident_and_arena(self):
        scheduler = self.scheduler(max_num_seqs=1)
        scheduler.add_request([3], request_id="a", max_new_tokens=16, tree_budget=3)
        scheduler.add_request([7], request_id="b", max_new_tokens=6, tree_budget=2)
        scheduler.step()
        self.assertTrue(self.runtime.block_manager.used_block_ids)
        scheduler.close()
        scheduler.close()
        self.assertFalse(self.runtime.requests)
        self.assertFalse(self.runtime.block_manager.used_block_ids)
        self.assertTrue(scheduler.is_finished())


class JetSpecContinuousEngineTest(ContinuousRuntimeFixture):
    def test_engine_routes_add_step_and_completion_without_heavy_constructor(self):
        engine = make_engine(self.runtime)
        engine.configure_jetspec("fake-draft", default_tree_budget=3)
        request_id = engine.add_request([3], SamplingParams(temperature=0, max_tokens=7), request_id="a")
        self.assertEqual(request_id, "a")
        emitted, completed = [], {}
        for _ in range(16):
            if engine.is_finished():
                break
            outputs, num_tokens = engine.step()
            self.assertIsInstance(num_tokens, int)
            for event in engine.last_step_info["events"]:
                if event["kind"] == "tokens":
                    emitted.extend(event["token_ids"])
            completed.update(outputs)
        self.assertTrue(engine.is_finished())
        self.assertEqual(completed["a"], list(range(4, 11)))
        self.assertEqual(emitted, completed["a"])
        self.serving = engine._jetspec_scheduler

    def test_serving_mode_gates_temperature_disable_and_manual_generation(self):
        engine = make_engine(self.runtime)
        engine.configure_jetspec("fake-draft", default_tree_budget=3)
        self.serving = engine._jetspec_scheduler
        with self.assertRaises(ValueError):
            engine.add_request([3], SamplingParams(temperature=1, max_tokens=7))
        engine.add_request([3], SamplingParams(temperature=0, max_tokens=7), request_id="a")
        with self.assertRaises(RuntimeError):
            engine.disable_jetspec()
        with self.assertRaises(RuntimeError):
            engine.generate_jetspec([3], "fake-draft")
        with self.assertRaises(RuntimeError):
            engine.generate_jetspec_batch([[3]], "fake-draft")
        engine.cancel_request("a")
        engine.step()
        engine.disable_jetspec()
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_configure_rejects_live_ordinary_or_explicit_packed_requests(self):
        engine = make_engine(self.runtime)
        engine.add_request([3], SamplingParams(temperature=0, max_tokens=7))
        with self.assertRaises(RuntimeError):
            engine.configure_jetspec("fake-draft", default_tree_budget=3)
        engine.scheduler.waiting.clear()
        request = self.runtime.create_request([3], request_id="manual", max_new_tokens=7, tree_budget=3)
        with self.assertRaises(RuntimeError):
            engine.configure_jetspec("fake-draft", default_tree_budget=3)
        self.runtime.cancel(request)

    def test_engine_exit_reclaims_serving_and_is_idempotent(self):
        engine = make_engine(self.runtime, max_num_seqs=1)
        self.serving = engine.configure_jetspec("fake-draft", default_tree_budget=3)
        engine.add_request([3], SamplingParams(temperature=0, max_tokens=12), request_id="a")
        engine.add_request([7], SamplingParams(temperature=0, max_tokens=12), request_id="b")
        engine.step()
        self.assertTrue(self.runtime.block_manager.used_block_ids)
        call = engine.model_runner.call
        engine.exit()
        engine.exit()
        self.assertFalse(self.runtime.block_manager.used_block_ids)
        self.assertFalse(self.runtime.requests)
        call.assert_called_once_with("exit")

    def test_configure_rejects_tp_and_graph_mode_without_loading_draft(self):
        engine = make_engine(self.runtime)
        engine.model_runner.world_size = 2
        with self.assertRaises(ValueError):
            engine.configure_jetspec("fake-draft", default_tree_budget=3)
        engine.model_runner.world_size = 1
        engine.model_runner.enforce_eager = False
        with self.assertRaises(ValueError):
            engine.configure_jetspec("fake-draft", default_tree_budget=3)

    def test_engine_failed_step_exposes_pending_error_report_and_exact_once_delivery(self):
        engine = make_engine(self.runtime, max_num_seqs=1)
        self.serving = engine.configure_jetspec("fake-draft", default_tree_budget=3)
        engine.add_request([3], SamplingParams(temperature=0, max_tokens=10), request_id="a")
        original = self.runtime._verify_batch
        self.runtime._verify_batch = Mock(side_effect=RuntimeError("engine injected verify failure"))
        with self.assertRaisesRegex(RuntimeError, "engine injected"):
            engine.step()
        self.assertIs(engine.last_step_info, self.serving.last_step)
        errors = [e for e in engine.last_step_info["events"] if e["kind"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["token_ids"], [4])
        self.assertFalse(engine.is_finished(), "undelivered error must remain observable")
        self.runtime._verify_batch = original
        outputs, num_tokens = engine.step()
        self.assertEqual(outputs, [("a", [4])])
        self.assertEqual(num_tokens, -1)
        self.assertTrue(engine.is_finished())
        self.assertEqual(engine.step(), ([], 0))


class JetSpecBlockingGenerateTest(ContinuousRuntimeFixture):
    def engine(self, *, max_num_seqs=1):
        engine = make_engine(self.runtime, max_num_seqs=max_num_seqs)
        self.serving = engine.configure_jetspec("fake-draft", default_tree_budget=3)
        return engine

    def assert_failed_invocation_cleaned(self):
        self.assertFalse(self.serving.requests)
        self.assertFalse(self.serving.pending_events)
        self.assertFalse(self.runtime.requests)
        self.assertFalse(self.runtime.block_manager.used_block_ids)
        self.assertFalse(self.runtime._closed)
        self.assertTrue(self.serving.is_finished())

    def test_success_preserves_input_order_mixed_caps_and_zero_output(self):
        engine = self.engine(max_num_seqs=2)
        params = [SamplingParams(temperature=0, max_tokens=cap) for cap in (7, 0, 2)]
        progress = Mock()
        with patch("nanovllm.engine.llm_engine.tqdm", return_value=progress):
            outputs = engine.generate([[3], [7], [11]], params, use_tqdm=False)
        self.assertEqual([output["token_ids"] for output in outputs],
                         [list(range(4, 11)), [], [12, 13]])
        self.assertEqual([output["text"] for output in outputs], ["4 5 6 7 8 9 10", "", "12 13"])
        self.assertTrue(engine.is_finished())
        self.assertFalse(self.runtime.requests)
        progress.close.assert_called_once()
        self.assertEqual(progress.update.call_count, 3)

    def test_sampling_count_mismatch_rejected_before_admission_and_progress_bar(self):
        engine = self.engine()
        with patch("nanovllm.engine.llm_engine.tqdm") as progress_factory:
            with self.assertRaisesRegex(ValueError, "sampling_params"):
                engine.generate([[3], [7]], [SamplingParams(temperature=0)], use_tqdm=False)
        progress_factory.assert_not_called()
        self.assert_failed_invocation_cleaned()

    def test_physical_capacity_terminal_error_is_not_success_and_report_is_preserved(self):
        self.replace_runtime(1)
        engine = self.engine()
        progress = Mock()
        with patch("nanovllm.engine.llm_engine.tqdm", return_value=progress):
            with self.assertRaisesRegex(RuntimeError, "KV capacity"):
                engine.generate([[3]], SamplingParams(temperature=0, max_tokens=4), use_tqdm=False)
        self.assert_failed_invocation_cleaned()
        errors = [event for event in engine.last_step_info["events"] if event["kind"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["token_ids"], [])
        self.assertIs(engine.last_step_info, self.serving.last_step)
        progress.close.assert_called_once()

    def test_partial_admission_validation_error_cleans_prior_submitted_id(self):
        engine = self.engine()
        progress = Mock()
        with patch("nanovllm.engine.llm_engine.tqdm", return_value=progress):
            with self.assertRaisesRegex(ValueError, "prompt"):
                engine.generate([[3], []], SamplingParams(temperature=0, max_tokens=7), use_tqdm=False)
        self.assert_failed_invocation_cleaned()
        self.assertIsNone(engine.last_step_info)
        progress.close.assert_called_once()
        # A failed wrapper must leave the configured adapter reusable.
        outputs = engine.generate([[3]], SamplingParams(temperature=0, max_tokens=4), use_tqdm=False)
        self.assertEqual(outputs[0]["token_ids"], [4, 5, 6, 7])

    def test_add_publication_before_raise_is_discovered_and_cleaned(self):
        engine = self.engine()
        injected = RuntimeError("add failed after publishing its ticket")
        original = engine.add_request

        def fail_after_add(*args, **kwargs):
            original(*args, **kwargs)
            raise injected

        progress = Mock()
        with patch.object(engine, "add_request", fail_after_add), patch(
            "nanovllm.engine.llm_engine.tqdm", return_value=progress
        ):
            with self.assertRaises(RuntimeError) as raised:
                engine.generate([[3]], SamplingParams(temperature=0, max_tokens=7), use_tqdm=False)
        self.assertIs(raised.exception, injected)
        self.assert_failed_invocation_cleaned()
        progress.close.assert_called_once()

    def test_precommit_error_preserves_original_exception_report_and_cleans_waiting(self):
        engine = self.engine()
        injected = RuntimeError("wrapper precommit verify failure")
        progress = Mock()
        with patch.object(self.runtime, "_verify_batch", side_effect=injected), patch(
            "nanovllm.engine.llm_engine.tqdm", return_value=progress
        ):
            with self.assertRaises(RuntimeError) as raised:
                engine.generate([[3], [7]], SamplingParams(temperature=0, max_tokens=10), use_tqdm=False)
        self.assertIs(raised.exception, injected)
        self.assert_failed_invocation_cleaned()
        self.assertIs(engine.last_step_info, self.serving.last_step)
        errors = [event for event in engine.last_step_info["events"] if event["kind"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["token_ids"], [4])
        progress.close.assert_called_once()

    def test_postcommit_error_keeps_committed_progress_report_and_cleans_all_own_ids(self):
        engine = self.engine()
        injected = RuntimeError("wrapper postcommit reporting failure")
        original = self.runtime.step

        def fail_after_step(*args, **kwargs):
            original(*args, **kwargs)
            raise injected

        progress = Mock()
        with patch.object(self.runtime, "step", fail_after_step), patch(
            "nanovllm.engine.llm_engine.tqdm", return_value=progress
        ):
            with self.assertRaises(RuntimeError) as raised:
                engine.generate([[3], [7]], SamplingParams(temperature=0, max_tokens=10), use_tqdm=False)
        self.assertIs(raised.exception, injected)
        self.assert_failed_invocation_cleaned()
        errors = [event for event in engine.last_step_info["events"] if event["kind"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["token_ids"], [4, 5, 6, 7])
        progress.close.assert_called_once()

    def test_cancelled_terminal_is_not_returned_as_success(self):
        engine = self.engine()
        original = engine.step

        def cancel_before_step():
            engine.cancel_request(next(iter(self.serving.requests)))
            return original()

        progress = Mock()
        with patch.object(engine, "step", cancel_before_step), patch(
            "nanovllm.engine.llm_engine.tqdm", return_value=progress
        ):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                engine.generate([[3]], SamplingParams(temperature=0, max_tokens=7), use_tqdm=False)
        self.assert_failed_invocation_cleaned()
        self.assertTrue(any(event["kind"] == "cancelled" for event in engine.last_step_info["events"]))
        progress.close.assert_called_once()

    def test_external_page_backpressure_is_bounded_and_does_not_release_foreign_leases(self):
        self.replace_runtime(3)
        engine = self.engine()
        held = self.runtime.block_manager.reserve_provisional(2)
        progress = Mock()
        try:
            with patch.object(engine, "step", wraps=engine.step) as step, patch(
                "nanovllm.engine.llm_engine.tqdm", return_value=progress
            ):
                with self.assertRaisesRegex(RuntimeError, "blocked"):
                    engine.generate([[3]], SamplingParams(temperature=0, max_tokens=4), use_tqdm=False)
                self.assertEqual(step.call_count, 1)
            self.assertTrue(engine.last_step_info["blocked"])
            self.assertFalse(self.serving.requests)
            self.assertFalse(self.serving.pending_events)
            self.assertFalse(self.runtime.requests)
            self.assertEqual(self.runtime.block_manager.used_block_ids, set(held))
            progress.close.assert_called_once()
        finally:
            self.runtime.block_manager.release_provisional(held)
        output = engine.generate([[3]], SamplingParams(temperature=0, max_tokens=4), use_tqdm=False)
        self.assertEqual(output[0]["token_ids"], [4, 5, 6, 7])

    def test_cleanup_and_progress_bar_failures_do_not_mask_original_exception(self):
        engine = self.engine()
        original_error = RuntimeError("original wrapper step failure")
        original_cancel = engine.cancel_request

        def fail_after_cancel(request_id):
            original_cancel(request_id)
            raise RuntimeError("cleanup raised after cancelling")

        progress = Mock()
        progress.close.side_effect = RuntimeError("progress close failed")
        with patch.object(engine, "step", side_effect=original_error), patch.object(
            engine, "cancel_request", fail_after_cancel
        ), patch("nanovllm.engine.llm_engine.tqdm", return_value=progress):
            with self.assertRaises(RuntimeError) as raised:
                engine.generate([[3], [7]], SamplingParams(temperature=0, max_tokens=7), use_tqdm=False)
        self.assertIs(raised.exception, original_error)
        notes = " ".join(getattr(original_error, "__notes__", []))
        self.assertIn("cleanup raised after cancelling", notes)
        self.assertIn("progress close failed", notes)
        self.assert_failed_invocation_cleaned()
        progress.close.assert_called_once()

    def test_cleanup_failure_keeps_live_ticket_ids_observable_and_original_error(self):
        engine = self.engine()
        original_error = RuntimeError("wrapper step failed before admission")
        progress = Mock()
        with patch.object(engine, "step", side_effect=original_error), patch.object(
            engine, "cancel_request", side_effect=RuntimeError("cancel could not release ticket")
        ), patch("nanovllm.engine.llm_engine.tqdm", return_value=progress):
            with self.assertRaises(RuntimeError) as raised:
                engine.generate([[3], [7]], SamplingParams(temperature=0, max_tokens=7), use_tqdm=False)
        self.assertIs(raised.exception, original_error)
        self.assertEqual(len(self.serving.requests), 2)
        self.assertEqual(set(self.serving.waiting), set(self.serving.requests))
        self.assertFalse(self.runtime.requests)
        self.assertIn("retains requests", " ".join(getattr(original_error, "__notes__", [])))
        for request_id in list(self.serving.requests):
            engine.cancel_request(request_id)
        self.serving.drain_events()
        self.assert_failed_invocation_cleaned()
        progress.close.assert_called_once()

    def test_initial_live_queue_is_rejected_without_cleanup_or_progress_bar(self):
        engine = self.engine()
        engine.add_request([3], SamplingParams(temperature=0, max_tokens=12), request_id="prior")
        engine.step()
        request = self.runtime.requests["prior"]
        before_output = list(request.output_ids)
        before_committed = request.state.committed.clone()
        before_pages = set(self.runtime.block_manager.used_block_ids)
        with patch("nanovllm.engine.llm_engine.tqdm") as progress_factory:
            with self.assertRaisesRegex(RuntimeError, "idle"):
                engine.generate([[7]], SamplingParams(temperature=0, max_tokens=7), use_tqdm=False)
        progress_factory.assert_not_called()
        self.assertIs(self.runtime.requests["prior"], request)
        self.assertEqual(set(self.serving.requests), {"prior"})
        self.assertEqual(request.output_ids, before_output)
        self.assertTrue(torch.equal(request.state.committed, before_committed))
        self.assertEqual(self.runtime.block_manager.used_block_ids, before_pages)


class OrdinarySchedulerContinuousBaselineTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler(SimpleNamespace(
            max_num_seqs=1, max_num_batched_tokens=32, eos=63,
            kvcache_block_size=256, num_kvcache_blocks=8,
        ))

    def tearDown(self):
        for sequence in list(self.scheduler.waiting) + list(self.scheduler.running):
            self.scheduler.cancel(sequence.seq_id)
        self.assertFalse(self.scheduler.block_manager.used_block_ids)

    def test_new_arrival_cannot_exceed_resident_cap_until_old_request_finishes(self):
        first = Sequence([1, 2, 3], SamplingParams(temperature=0, max_tokens=2))
        second = Sequence([7], SamplingParams(temperature=0, max_tokens=2))
        self.scheduler.add(first)
        scheduled, is_prefill = self.scheduler.schedule()
        self.assertTrue(is_prefill)
        self.scheduler.postprocess(scheduled, [4], is_prefill)
        self.assertEqual(list(self.scheduler.running), [first])
        self.scheduler.add(second)
        scheduled, is_prefill = self.scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [first])
        self.assertEqual(list(self.scheduler.running), [first])
        self.assertEqual(list(self.scheduler.waiting), [second])
        self.assertFalse(second.block_table)
        self.scheduler.postprocess(scheduled, [5], is_prefill)
        self.assertTrue(first.is_finished)
        scheduled, is_prefill = self.scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [second])
        self.assertEqual(len(self.scheduler.running), 1)

    def test_cancel_queued_and_running_sequences_reclaims_all_owned_pages(self):
        queued = Sequence([7], SamplingParams(temperature=0, max_tokens=8))
        self.scheduler.add(queued)
        self.assertIs(self.scheduler.cancel(queued.seq_id), queued)
        self.assertEqual(queued.status, SequenceStatus.FINISHED)
        self.assertFalse(self.scheduler.block_manager.used_block_ids)
        running = Sequence([3] * 257, SamplingParams(temperature=0, max_tokens=8))
        self.scheduler.max_num_batched_tokens = 512
        self.scheduler.add(running)
        scheduled, is_prefill = self.scheduler.schedule()
        self.scheduler.postprocess(scheduled, [4], is_prefill)
        self.assertEqual(len(running.block_table), 2)
        self.assertEqual(len(self.scheduler.block_manager.used_block_ids), 2)
        self.assertIs(self.scheduler.cancel(running.seq_id), running)
        self.assertFalse(running.block_table)
        self.assertFalse(self.scheduler.block_manager.used_block_ids)
        self.assertIsNone(self.scheduler.cancel(running.seq_id))
        self.assertIsNone(self.scheduler.cancel(-999))
        self.assertTrue(self.scheduler.is_finished())
