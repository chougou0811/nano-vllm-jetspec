"""CPU-only independently specified metrics/ledger tests; no model imports."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import jetspec_final_matched as bench


class FinalMatchedMetricTests(unittest.TestCase):
    def spec(self, name="r0", cap=3):
        return {"request_id": name, "prompt": [1], "max_tokens": cap,
                "tree_budget": 1, "arrival_s": .05}

    def test_burst_tpot_uses_output_tokens_not_deliveries(self):
        ledger = bench.Ledger([self.spec()])
        ledger.submit("r0", .1)
        ledger.consume([{"request_id": 91, "kind": "tokens", "token_ids": [1, 2]}], .2, {91: "r0"})
        ledger.consume([{"request_id": 91, "kind": "tokens", "token_ids": [3]},
                        {"request_id": 91, "kind": "finished", "token_ids": [1, 2, 3]}], .6, {91: "r0"})
        result = ledger.summary()
        row = result["requests"][0]
        for key, expected in {"offered_ttft_s": .15, "submitted_ttft_s": .1,
                              "offered_e2e_s": .55, "submitted_e2e_s": .5,
                              "submission_lag_s": .05, "delivery_tpot_s": .2}.items():
            self.assertAlmostEqual(row[key], expected)
        self.assertEqual(row["deliveries"], [{"at_s": .2, "count": 2}, {"at_s": .6, "count": 1}])
        self.assertAlmostEqual(result["unique_nonempty_batch_event_gaps_s"][0], .4)

    def test_same_step_bursts_keep_zero_time_and_one_batch_event(self):
        ledger = bench.Ledger([self.spec("r0", 2), self.spec("r1", 1)])
        for name in ("r0", "r1"):
            ledger.submit(name, .1)
        ledger.consume([{"request_id": "r0", "kind": "tokens", "token_ids": [1]},
                        {"request_id": "r0", "kind": "tokens", "token_ids": [2]},
                        {"request_id": "r1", "kind": "tokens", "token_ids": [9]},
                        {"request_id": "r0", "kind": "finished", "token_ids": [1, 2]},
                        {"request_id": "r1", "kind": "finished", "token_ids": [9]}],
                       .2, {"r0": "r0", "r1": "r1"})
        result = ledger.summary()
        self.assertEqual(result["requests"][0]["delivery_tpot_s"], 0)
        self.assertIsNone(result["requests"][1]["delivery_tpot_s"])
        self.assertEqual(result["nonempty_batch_delivery_times_s"], [.2])
        self.assertEqual(result["unique_nonempty_batch_event_gaps_s"], [])
        self.assertEqual([d["count"] for d in result["requests"][0]["deliveries"]], [2])
        self.assertEqual([d["count"] for d in result["requests"][0]["delivery_events"]], [1, 1])
        self.assertEqual(result["requests"][0]["delivery_event_gaps_s"], [])

    def test_exactly_once_terminal_and_caps_are_hard_gates(self):
        ledger = bench.Ledger([self.spec(cap=1)])
        ledger.submit("r0", .1)
        with self.assertRaisesRegex(AssertionError, "cap exceeded"):
            ledger.consume([{"request_id": 1, "kind": "tokens", "token_ids": [1, 2]}], .2, {1: "r0"})
        ledger = bench.Ledger([self.spec(cap=1)])
        ledger.submit("r0", .1)
        ledger.consume([{"request_id": 1, "kind": "tokens", "token_ids": [1]}], .2, {1: "r0"})
        with self.assertRaisesRegex(AssertionError, "history disagrees"):
            ledger.consume([{"request_id": 1, "kind": "finished", "token_ids": [2]}], .3, {1: "r0"})
        ledger.consume([{"request_id": 1, "kind": "finished", "token_ids": [1]}], .3, {1: "r0"})
        with self.assertRaisesRegex(AssertionError, "duplicate terminal"):
            ledger.consume([{"request_id": 1, "kind": "finished", "token_ids": [1]}], .4, {1: "r0"})

    def test_fixed_three_run_statistics_and_peak_max(self):
        # Explicit expected values: per-run p95 = 95,59,86 => median86.
        # This is NOT pooled p95, which would be97.5 for these six values.
        samples = []
        latency_values = ((0, 100), (40, 60), (10, 90))
        for mode, throughputs in (("ordinary_ar", (10, 40, 20)), ("jetspec", (60, 30, 50))):
            for i, throughput in enumerate(throughputs):
                samples.append({"mode": mode, "concurrency": 1, "output_cap_scale": 128,
                    "repeat": i,
                    "workload_sha256": "same", "actual_output_tokens": 100,
                    "tokens_per_second": throughput, "wall_s": 100 / throughput,
                    "request_metrics": {key: bench.distribution(latency_values[i]) for key in bench.REQUEST_METRICS},
                    "batch_event_gap_distribution_s": bench.distribution((i + 1, i + 2)),
                    "per_request_delivery_gap_distribution_s": bench.distribution((i + 1, i + 2)),
                    "peak_used_pages": (2, 5, 3)[i], "peak_reserved_kv_slots": (512, 1280, 768)[i],
                    "peak_gpu_allocated_bytes": (100, 150, 120)[i],
                    "peak_gpu_reserved_bytes": (200, 250, 220)[i],
                    "peak_gpu_allocated_delta_bytes": (10, 50, 20)[i],
                    "peak_gpu_reserved_delta_bytes": (20, 80, 30)[i],
                    "mean_effective_output_block_tokens_per_packed_verify_call": (4, 8, 6)[i],
                    "mean_effective_output_block_tokens_per_verified_request": (2, 4, 3)[i]})
        case = bench.matched_summary(samples)["cases"][0]
        self.assertEqual(case["jetspec_over_ar_median_throughput_ratio"], 2.5)
        ar, jet = case["variants"]
        self.assertEqual(ar["tokens_per_second"]["p50"], 20)
        self.assertEqual(jet["tokens_per_second"]["p50"], 50)
        self.assertEqual(ar["median_of_sample_request_metrics"]["offered_ttft_s"]["p95"], 86)
        self.assertEqual(ar["max_peak_used_pages"], 5)
        self.assertEqual(ar["max_peak_gpu_allocated_bytes"], 150)
        self.assertEqual(jet["median_effective_output_block_tokens_per_packed_verify_call"], 6)
        self.assertEqual(jet["median_effective_output_block_tokens_per_verified_request"], 3)
        incomplete = deepcopy(samples[:2] + samples[3:])
        with self.assertRaisesRegex(AssertionError, "three timed"):
            bench.matched_summary(incomplete)

    def test_workload_dimensions_and_balanced_orders(self):
        class Tokenizer:
            def encode(self, text):
                return [len(text), 3, 5]
        manifest = bench.build_manifest(Tokenizer())
        self.assertEqual(len(manifest["cases"]), 6)
        case = manifest["cases"][2]
        self.assertEqual((case["concurrency"], case["output_cap_scale"]), (4, 128))
        self.assertEqual([len(s["prompt"]) for s in case["specs"]], [128] * 4 + [1024, 2048, 1024, 2048])
        self.assertEqual([s["max_tokens"] for s in case["specs"]], [128, 128, 64, 32] * 2)
        self.assertEqual([s["tree_budget"] for s in case["specs"]], [63, 31, 47, 63, 31, 47, 63, 31])
        self.assertEqual([s["arrival_s"] for s in case["specs"]], [0] * 4 + [.02, .04, .06, .08])
        self.assertTrue(all(type(s["arrival_s"]) is int for s in case["specs"][:4]))
        self.assertEqual(bench.mode_order(0, 0), ["ordinary_ar", "jetspec"])
        self.assertEqual(bench.mode_order(1, 0), ["jetspec", "ordinary_ar"])
        self.assertEqual(bench.mode_order(0, 1), ["jetspec", "ordinary_ar"])

    def test_observer_catches_prefix_cache_reactivation_without_fresh_allocate(self):
        class Manager:
            used_block_ids = set()
            def _allocate_block(self):
                self.used_block_ids.add(0)
                return 0
            def allocate(self):
                self.used_block_ids.update((1, 2, 3))  # free-prefix-cache path bypasses _allocate_block
        manager = Manager()
        with bench.AllocationObserver(manager) as observer:
            manager._allocate_block()
            manager.allocate()
        self.assertEqual(observer.peak_blocks, 4)

    def test_prepare_uses_real_runner_config_location_and_rebinds_cached_allocator(self):
        class Manager:
            def __init__(self, count, size):
                self.blocks = [SimpleNamespace(block_id=i, ref_count=0, hash=-1, token_ids=[]) for i in range(count)]
                self.block_size, self.hash_to_block_id = size, {}
                self.used_block_ids, self.free_block_ids = set(), list(range(count))
        manager = Manager(9, 256)
        runtime = SimpleNamespace(block_manager=manager, requests={}, prefills={}, _active_transaction=None,
            arena=SimpleNamespace(block_manager=manager, blocks=[], active=False, _batch_transaction=None),
            release_idle_scratch=lambda: 0)
        engine = SimpleNamespace(scheduler=SimpleNamespace(block_manager=manager, waiting=[], running=[], max_num_seqs=8),
            model_runner=SimpleNamespace(config=SimpleNamespace(max_num_seqs=8)),
            is_finished=lambda: True, disable_jetspec=lambda: None)
        # No engine.config or _jetspec_scheduler attributes, just like a fresh LLM.
        with patch.dict(sys.modules, {"nanovllm.engine.block_manager": SimpleNamespace(BlockManager=Manager)}):
            cold = bench.prepare_sample(engine, runtime, "draft", 4, "ordinary_ar")
        self.assertEqual(engine.model_runner.config.max_num_seqs, 4)
        self.assertEqual(engine.scheduler.max_num_seqs, 4)
        self.assertIs(runtime.block_manager, engine.scheduler.block_manager)
        self.assertIs(runtime.arena.block_manager, engine.scheduler.block_manager)
        self.assertIsNot(manager, engine.scheduler.block_manager)
        self.assertEqual(cold["used_blocks"], 0)
        self.assertEqual(cold["hash_entries"], 0)


if __name__ == "__main__":
    unittest.main()
