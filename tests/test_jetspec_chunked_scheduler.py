"""Serving gates for partial initial prefill and partial recomputation.

The deterministic model removes cross-shape BF16 effects; the real scheduler,
runtime, page allocator, feature state and commit transactions remain in use.
"""
import unittest
from unittest.mock import Mock

from nanovllm.engine.jetspec_scheduler import JetSpecScheduler
from nanovllm.sampling_params import SamplingParams
from tests.test_jetspec_continuous import ContinuousRuntimeFixture, make_engine


class ChunkedSchedulerTest(ContinuousRuntimeFixture):
    def scheduler(self, **kwargs):
        kwargs.setdefault("max_num_seqs", 2)
        kwargs.setdefault("max_prefill_tokens", 8)
        kwargs.setdefault("prefill_chunk_size", 4)
        self.serving = JetSpecScheduler(self.runtime, **kwargs)
        return self.serving

    def collect(self, reports):
        emitted, terminal = {}, {}
        for report in reports:
            self.assertLessEqual(report["prefill_tokens"], self.serving.max_prefill_tokens)
            self.assertLessEqual(report["running_count"] + report["prefilling_count"],
                                 self.serving.max_num_seqs)
            for chunk in report["prefill_chunks"]:
                self.assertLessEqual(chunk["end"] - chunk["start"], self.serving.prefill_chunk_size)
            for event in report["events"]:
                rid = event["request_id"]
                if event["kind"] == "tokens":
                    emitted.setdefault(rid, []).extend(event["token_ids"])
                    self.assertEqual(len(emitted[rid]), event["output_length"])
                elif event["kind"] == "error":
                    self.fail(event["reason"])
                elif event["kind"] in ("finished", "cancelled"):
                    self.assertNotIn(rid, terminal)
                    terminal[rid] = event["token_ids"]
                    self.assertEqual(emitted.get(rid, []), terminal[rid])
        return emitted, terminal

    def test_long_prompt_uses_multiple_steps_without_early_anchor_or_draft(self):
        scheduler = self.scheduler(max_prefill_tokens=4)
        scheduler.add_request([3] * 11, request_id="long", max_new_tokens=1, tree_budget=3)
        for expected in (4, 8):
            report = scheduler.step()
            self.assertFalse(report["blocked"])
            self.assertIsNone(report["verification"])
            self.assertEqual(report["events"], [])
            self.assertNotIn("long", self.runtime.requests)
            self.assertEqual(scheduler.requests["long"].prefill.processed_tokens, expected)
            self.assertEqual(scheduler.requests["long"].prefill.target_hidden.shape[1], expected)
        final = scheduler.step()
        self.assertEqual(final["prefill_tokens"], 3)
        self.assertEqual([e["token_ids"] for e in final["events"]], [[4], [4]])
        self.assertTrue(scheduler.is_finished())
        self.assertFalse(self.runtime.prefills)
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_dynamic_long_arrival_does_not_stop_resident_packed_verification(self):
        scheduler = self.scheduler(max_num_seqs=3, max_prefill_tokens=4)
        scheduler.add_request([3], request_id="a", max_new_tokens=25, tree_budget=3)
        reports = [scheduler.step()]
        scheduler.add_request([7] * 13, request_id="b", max_new_tokens=7, tree_budget=2)
        scheduler.add_request([11] * 5, request_id="c", max_new_tokens=5, tree_budget=1)
        seen_partial_decode = 0
        while not scheduler.is_finished():
            report = scheduler.step()
            reports.append(report)
            if report["prefill_chunks"] and report["verification"]:
                seen_partial_decode += 1
                # A partial is not included in the same step's resident verify.
                partial_ids = {c["request_id"] for c in report["prefill_chunks"]}
                self.assertFalse(partial_ids & set(report["verification"]["request_ids"]))
        self.assertGreaterEqual(seen_partial_decode, 3)
        _, terminal = self.collect(reports)
        self.assertEqual(terminal, {"a": list(range(4, 29)), "b": list(range(8, 15)),
                                    "c": list(range(12, 17))})

    def test_round_robin_prefill_under_one_quantum_total_budget(self):
        scheduler = self.scheduler(max_prefill_tokens=4)
        for name in ("a", "b"):
            scheduler.add_request([3] * 9, request_id=name, max_new_tokens=1, tree_budget=3)
        reports = self.drain(scheduler)
        self.assertEqual([c["request_id"] for r in reports for c in r["prefill_chunks"]],
                         ["a", "b", "a", "b", "a", "b"])
        self.collect(reports)

    def test_cancel_partial_returns_no_tokens_and_releases_pages(self):
        scheduler = self.scheduler()
        scheduler.add_request([3] * 17, request_id="partial", max_new_tokens=8, tree_budget=3)
        scheduler.step()
        self.assertTrue(self.runtime.block_manager.used_block_ids)
        event = scheduler.cancel("partial")
        self.assertEqual(event["token_ids"], [])
        self.assertFalse(self.runtime.prefills)
        self.assertFalse(self.runtime.block_manager.used_block_ids)
        self.assertIsNone(scheduler.cancel("partial"))
        report = scheduler.step()
        self.assertEqual([e["kind"] for e in report["events"]], ["cancelled"])
        self.assertTrue(scheduler.is_finished())

    def test_chunk_failure_releases_partial_and_preserves_other_ticket(self):
        scheduler = self.scheduler(max_num_seqs=1)
        scheduler.add_request([3] * 9, request_id="failed", max_new_tokens=5, tree_budget=3)
        scheduler.add_request([7], request_id="next", max_new_tokens=2, tree_budget=1)
        scheduler.step()
        original = self.runtime.target.model.forward_dense
        self.runtime.target.model.forward_dense = Mock(side_effect=RuntimeError("chunk failure"))
        try:
            with self.assertRaisesRegex(RuntimeError, "chunk failure"):
                scheduler.step()
        finally:
            self.runtime.target.model.forward_dense = original
        self.assertFalse(self.runtime.prefills)
        self.assertFalse(self.runtime.block_manager.used_block_ids)
        errors = scheduler.drain_events()
        self.assertEqual([(e["kind"], e["token_ids"]) for e in errors], [("error", [])])
        self.assertIn("next", scheduler.requests)
        self.collect(self.drain(scheduler))

    def test_error_after_final_promotion_discovers_ready_state_and_emits_once(self):
        scheduler = self.scheduler()
        scheduler.add_request([3] * 5, request_id="a", max_new_tokens=5, tree_budget=3)
        scheduler.step()
        original = self.runtime.prefill_step
        def after_publish(context, count):
            request = original(context, count)
            self.assertIsNotNone(request)
            raise RuntimeError("after READY publication")
        self.runtime.prefill_step = after_publish
        try:
            with self.assertRaisesRegex(RuntimeError, "after READY publication"):
                scheduler.step()
        finally:
            self.runtime.prefill_step = original
        events = scheduler.drain_events()
        self.assertEqual([(e["kind"], e["token_ids"]) for e in events],
                         [("tokens", [4]), ("error", [4])])
        self.assertFalse(self.runtime.prefills)
        self.assertFalse(self.runtime.requests)
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_partial_release_failure_keeps_ownership_for_cancel_retry(self):
        from unittest.mock import patch
        scheduler = self.scheduler()
        scheduler.add_request([3] * 10, request_id="a", max_new_tokens=5, tree_budget=3)
        scheduler.step()
        partial = scheduler.requests["a"].prefill
        before = set(self.runtime.block_manager.used_block_ids)
        with patch.object(self.runtime.block_manager, "release_provisional",
                          side_effect=RuntimeError("partial release failure")):
            with self.assertRaisesRegex(RuntimeError, "partial release failure"):
                scheduler.cancel("a")
        self.assertIs(scheduler.requests["a"].prefill, partial)
        self.assertEqual(self.runtime.block_manager.used_block_ids, before)
        self.assertEqual(scheduler.drain_events(), [])
        self.assertEqual(scheduler.cancel("a")["kind"], "cancelled")
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_each_scheduler_step_executes_at_most_one_packed_verification(self):
        scheduler = self.scheduler(max_prefill_tokens=4)
        scheduler.add_request([3], request_id="a", max_new_tokens=7, tree_budget=3)
        scheduler.add_request([7] * 9, request_id="b", max_new_tokens=4, tree_budget=2)
        original = self.runtime.step
        counter = Mock(wraps=original)
        self.runtime.step = counter
        try:
            while not scheduler.is_finished():
                previous = counter.call_count
                scheduler.step()
                self.assertLessEqual(counter.call_count - previous, 1)
        finally:
            self.runtime.step = original

    def test_chunked_recompute_exceeds_step_budget_without_reemission(self):
        scheduler = self.scheduler(max_prefill_tokens=4)
        scheduler.add_request([3] * 10, request_id="a", max_new_tokens=12, tree_budget=3)
        reports = []
        while "a" not in self.runtime.requests:
            reports.append(scheduler.step())
        ticket = scheduler.requests["a"]
        old_output = list(ticket.output_ids)
        self.assertTrue(old_output)
        victims = []
        scheduler._preempt(ticket, victims)
        prefix = len(ticket.snapshot["committed_tokens"]) - 1
        self.assertGreater(prefix, scheduler.max_prefill_tokens)
        reports.extend(self.drain(scheduler))
        records = [c for r in reports for c in r["prefill_chunks"] if c["is_recompute"]]
        self.assertGreater(len(records), 1)
        self.assertEqual(sum(c["end"] - c["start"] for c in records), prefix)
        self.assertEqual(sum("a" in r["resumed_ids"] for r in reports), 1)
        _, terminal = self.collect(reports)
        self.assertEqual(terminal["a"], list(range(4, 16)))

    def test_cancel_mid_recompute_preserves_previously_delivered_tokens(self):
        scheduler = self.scheduler(max_prefill_tokens=4)
        scheduler.add_request([3] * 8, request_id="a", max_new_tokens=12, tree_budget=3)
        reports = [scheduler.step(), scheduler.step()]
        ticket = scheduler.requests["a"]
        output = list(ticket.output_ids)
        scheduler._preempt(ticket, [])
        recompute = scheduler.step()
        self.assertEqual(recompute["events"], [])
        self.assertTrue(recompute["prefill_chunks"][0]["is_recompute"])
        self.assertEqual(scheduler.cancel("a")["token_ids"], output)
        reports.extend([recompute, scheduler.step()])
        _, terminal = self.collect(reports)
        self.assertEqual(terminal["a"], output)
        self.assertFalse(self.runtime.prefills)

    def test_allocator_driven_preemption_resumes_prefix_larger_than_step_budget(self):
        self.replace_runtime(3)
        scheduler = self.scheduler(max_prefill_tokens=128, prefill_chunk_size=64)
        scheduler.add_request([3] * 250, request_id="a", max_new_tokens=13, tree_budget=3)
        scheduler.add_request([7] * 250, request_id="b", max_new_tokens=10, tree_budget=3)
        reports = self.drain(scheduler, max_steps=80)
        preempted = {rid for r in reports for rid in r["preempted_ids"]}
        resumed = {rid for r in reports for rid in r["resumed_ids"]}
        self.assertTrue(preempted, "must trigger actual allocator-driven recompute")
        self.assertTrue(preempted <= resumed)
        recompute = [c for r in reports for c in r["prefill_chunks"] if c["is_recompute"]]
        self.assertTrue(recompute)
        self.assertTrue(all(c["total_tokens"] > scheduler.max_prefill_tokens for c in recompute))
        _, terminal = self.collect(reports)
        self.assertEqual(terminal, {"a": list(range(4, 17)), "b": list(range(8, 18))})

    def test_randomized_small_pool_lifecycles_have_no_unowned_deadlock(self):
        import random
        rng = random.Random(412)
        for _ in range(12):
            if self.serving is not None:
                self.serving.close()
            self.replace_runtime(rng.choice((4, 5, 6)))
            scheduler = self.scheduler(max_num_seqs=3,
                max_prefill_tokens=rng.choice((32, 128, 512)),
                prefill_chunk_size=rng.choice((31, 64, 256)))
            for index in range(5):
                scheduler.add_request([3 + index] * rng.choice((1, 249, 256, 257, 510)),
                    request_id=index, max_new_tokens=rng.randrange(2, 14),
                    tree_budget=rng.randrange(1, 4))
            reports = self.drain(scheduler, max_steps=300)
            self.assertFalse(any(r["blocked"] for r in reports),
                             "no external leases and singleton-feasible prefixes must progress")
            self.collect(reports)

    def test_retried_long_prefill_cannot_starve_under_sustained_short_arrivals(self):
        self.replace_runtime(4)
        scheduler = self.scheduler(max_num_seqs=2, max_prefill_tokens=512, prefill_chunk_size=256)
        scheduler.add_request([3] * 550, request_id="old", max_new_tokens=5, tree_budget=3)
        old_output, old_finished, partial_evictions = [], False, 0
        for tick in range(100):
            scheduler.add_request([7] * 249, request_id=f"new-{tick}",
                                  max_new_tokens=20, tree_budget=3)
            report = scheduler.step()
            partial_evictions += report["prefill_preempted_ids"].count("old")
            for event in report["events"]:
                if event["request_id"] != "old":
                    continue
                self.assertNotEqual(event["kind"], "error")
                if event["kind"] == "tokens":
                    old_output.extend(event["token_ids"])
                elif event["kind"] == "finished":
                    self.assertEqual(event["token_ids"], old_output)
                    old_finished = True
            if old_finished:
                break
        self.assertTrue(partial_evictions, "must actually exercise the starvation risk")
        self.assertTrue(old_finished, "retried old prefix starved behind sustained new arrivals")
        self.assertEqual(old_output, [4, 5, 6, 7, 8])
        scheduler.close()
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_partial_prefix_pressure_reclaims_younger_partial_and_finishes(self):
        self.replace_runtime(4)
        scheduler = self.scheduler(max_prefill_tokens=512, prefill_chunk_size=256)
        scheduler.add_request([3] * 550, request_id="a", max_new_tokens=5, tree_budget=3)
        scheduler.add_request([7] * 550, request_id="b", max_new_tokens=4, tree_budget=2)
        reports = self.drain(scheduler, max_steps=32)
        self.assertTrue(any(r["prefill_preempted_ids"] for r in reports))
        self.assertLessEqual(max(r["capacity"]["allocator_used_blocks"] for r in reports), 4)
        _, terminal = self.collect(reports)
        self.assertEqual(terminal, {"a": [4, 5, 6, 7, 8], "b": [8, 9, 10, 11]})

    def test_held_pages_backpressure_is_retryable_and_not_false_progress(self):
        self.replace_runtime(3)
        scheduler = self.scheduler()
        held = self.runtime.block_manager.reserve_provisional(3)
        scheduler.add_request([3] * 9, request_id="a", max_new_tokens=3, tree_budget=3)
        self.assertTrue(scheduler.step()["blocked"])  # Registration is not GPU progress.
        blocked = scheduler.step()
        self.assertTrue(blocked["blocked"])
        self.assertEqual(blocked["prefill_tokens"], 0)
        self.assertEqual(blocked["blocked_reason"], "waiting_for_kv_prefill")
        self.runtime.block_manager.release_provisional(held)
        self.collect(self.drain(scheduler))

    def test_chunking_does_not_admit_physically_impossible_prefix(self):
        self.replace_runtime(2)
        scheduler = self.scheduler()
        scheduler.add_request([3] * 512, request_id="too-large", max_new_tokens=4, tree_budget=3)
        report = scheduler.step()
        self.assertEqual(report["events"][0]["kind"], "error")
        self.assertIn("cannot fit", report["events"][0]["reason"])
        self.assertFalse(self.runtime.block_manager.used_block_ids)
        self.assertTrue(scheduler.is_finished())

    def test_eos_zero_cap_and_partial_close(self):
        self.runtime.eos_token_ids = {5}
        scheduler = self.scheduler()
        scheduler.add_request([4] * 5, request_id="eos", max_new_tokens=8, tree_budget=3)
        scheduler.add_request([4] * 5, request_id="ignore", max_new_tokens=4, tree_budget=3, ignore_eos=True)
        scheduler.add_request([3] * 20, request_id="zero", max_new_tokens=0, tree_budget=3)
        _, terminal = self.collect(self.drain(scheduler))
        self.assertEqual(terminal, {"eos": [5], "ignore": [5, 6, 7, 8], "zero": []})
        scheduler.add_request([3] * 20, request_id="partial", max_new_tokens=5, tree_budget=3)
        scheduler.step()
        scheduler.close()
        self.assertFalse(self.runtime.prefills)
        self.assertFalse(self.runtime.block_manager.used_block_ids)
        self.assertTrue(scheduler.is_finished())

    def test_nonchunked_reference_remains_explicit(self):
        scheduler = self.scheduler(enable_chunked_prefill=False)
        with self.assertRaisesRegex(ValueError, "nonchunked"):
            scheduler.add_request([3] * 9, max_new_tokens=1, tree_budget=3)

    def test_engine_generate_default_handles_partial_steps_as_progress(self):
        engine = make_engine(self.runtime)
        self.serving = engine.configure_jetspec("fake-draft", default_tree_budget=3,
            max_prefill_tokens=4, prefill_chunk_size=3)
        self.assertTrue(self.serving.enable_chunked_prefill)
        outputs = engine.generate([[3] * 10, [7] * 9],
            SamplingParams(temperature=0, max_tokens=6), use_tqdm=False)
        self.assertEqual([r["token_ids"] for r in outputs],
                         [list(range(4, 10)), list(range(8, 14))])
        self.assertFalse(self.runtime.prefills)
        engine.disable_jetspec()
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_engine_refuses_ordinary_serving_with_manual_partial(self):
        partial = self.runtime.begin_prefill([3] * 10, request_id="manual", max_new_tokens=4, tree_budget=3)
        engine = make_engine(self.runtime)
        with self.assertRaises(RuntimeError):
            engine.add_request([3], SamplingParams(temperature=0, max_tokens=1))
        with self.assertRaises(RuntimeError):
            engine.configure_jetspec("fake-draft")
        self.runtime.cancel_prefill(partial)


if __name__ == "__main__":
    unittest.main()
