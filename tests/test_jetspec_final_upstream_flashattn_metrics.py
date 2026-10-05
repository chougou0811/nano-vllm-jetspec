"""CPU-only tests for pristine step observations and reused benchmark metrics."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import jetspec_final_upstream_flashattn as bench


def sequence(identifier=7):
    return SimpleNamespace(seq_id=identifier, num_prompt_tokens=2, num_tokens=2,
                           token_ids=[9, 8], is_finished=False)


class UpstreamBenchmarkMetricTests(unittest.TestCase):
    def test_native_observer_reads_deltas_without_repeating_history(self):
        observer = bench.NativeDeliveryObserver()
        seq = sequence()
        observer.add("r0", seq)
        self.assertEqual(observer.events([]), [])
        seq.token_ids.append(11)
        seq.num_tokens = 3
        self.assertEqual(observer.events([]), [
            {"request_id": "r0", "kind": "tokens", "token_ids": [11]}])
        self.assertEqual(observer.events([]), [])
        seq.token_ids.extend([12, 13])
        seq.num_tokens = 5
        seq.is_finished = True
        self.assertEqual(observer.events([(7, [11, 12, 13])]), [
            {"request_id": "r0", "kind": "tokens", "token_ids": [12, 13]},
            {"request_id": "r0", "kind": "finished", "token_ids": [11, 12, 13]}])
        self.assertEqual(observer.events([]), [])
        with self.assertRaisesRegex(AssertionError, "repeated native terminal"):
            observer.events([(7, [11, 12, 13])])

    def test_native_observer_rejects_bad_terminal_and_duplicate_native_ids(self):
        observer = bench.NativeDeliveryObserver()
        seq = sequence()
        observer.add("r0", seq)
        with self.assertRaisesRegex(AssertionError, "duplicate native sequence ID"):
            observer.add("r1", sequence())
        seq.token_ids.append(11)
        seq.num_tokens = 3
        seq.is_finished = True
        with self.assertRaisesRegex(AssertionError, "disagree"):
            observer.events([(7, [22])])

    def test_read_only_delivery_times_use_same_burst_tpot_contract(self):
        spec = {"request_id": "r0", "prompt": [9, 8], "max_tokens": 3,
                "tree_budget": 63, "arrival_s": .05}
        ledger = bench.previous.Ledger([spec])
        observer = bench.NativeDeliveryObserver()
        seq = sequence()
        observer.add("r0", seq)
        ledger.submit("r0", .1)
        seq.token_ids.extend([11, 12])
        seq.num_tokens = 4
        ledger.consume(observer.events([]), .2, {"r0": "r0"})
        seq.token_ids.append(13)
        seq.num_tokens = 5
        seq.is_finished = True
        ledger.consume(observer.events([(7, [11, 12, 13])]), .6, {"r0": "r0"})
        row = ledger.summary()["requests"][0]
        self.assertAlmostEqual(row["delivery_tpot_s"], .2)
        self.assertAlmostEqual(row["offered_ttft_s"], .15)
        self.assertAlmostEqual(row["offered_e2e_s"], .55)

    def test_manifest_is_exact_existing_hash_not_regenerated(self):
        class Tokenizer:
            def encode(self, _):
                return [3, 5, 7]
        manifest = bench.previous.build_manifest(Tokenizer())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(bench.load_manifest(path), manifest)
            manifest["cases"][0]["specs"][0]["max_tokens"] += 1
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(AssertionError, "content/hash mismatch"):
                bench.load_manifest(path)

    def test_flash_qualification_counts_original_wrappers_then_removes_profiler(self):
        def varlen():
            return 11
        def kvcache():
            return 12
        attention = SimpleNamespace(flash_attn_varlen_func=varlen, flash_attn_with_kvcache=kvcache)
        with bench.FlashCallEvidence(attention) as evidence:
            self.assertEqual(varlen(), 11)
            self.assertEqual(kvcache(), 12)
            varlen()
        self.assertIsNone(sys.getprofile())
        self.assertIs(attention.flash_attn_varlen_func, varlen)
        result = evidence.result()
        self.assertEqual(result["counts"], {"flash_attn_varlen_func": 2, "flash_attn_with_kvcache": 1})
        self.assertTrue(result["profiler_removed_for_timed_samples"])

    def test_native_cold_reset_preserves_allocator_geometry_and_case_setting(self):
        class Manager:
            def __init__(self, count, size):
                self.blocks = [SimpleNamespace(ref_count=0) for _ in range(count)]
                self.block_size = size
                self.used_block_ids = set()
                self.free_block_ids = list(range(count))
                self.hash_to_block_id = {}
        manager = Manager(91, 256)
        manager.hash_to_block_id[100] = 10
        engine = SimpleNamespace(is_finished=lambda: True,
            scheduler=SimpleNamespace(waiting=[], running=[], block_manager=manager, max_num_seqs=8),
            model_runner=SimpleNamespace(config=SimpleNamespace(max_num_seqs=8)))
        with patch.dict(sys.modules, {"nanovllm.engine.block_manager": SimpleNamespace(BlockManager=Manager)}):
            result = bench.prepare_upstream(engine, 4)
        self.assertEqual(result["free_blocks"], 91)
        self.assertEqual(result["hash_entries"], 0)
        self.assertEqual(engine.scheduler.block_manager.block_size, 256)
        self.assertEqual(engine.scheduler.max_num_seqs, 4)
        self.assertEqual(engine.model_runner.config.max_num_seqs, 4)
        self.assertIsNot(engine.scheduler.block_manager, manager)

    def test_combined_statistics_are_medians_not_best_or_temporally_paired(self):
        def samples(mode, throughputs, pages):
            rows = []
            for i, throughput in enumerate(throughputs):
                dist = bench.previous.distribution((i + 1, i + 2))
                rows.append({"mode": mode, "concurrency": 8, "output_cap_scale": 512,
                    "repeat": i, "workload_sha256": "identical", "actual_output_tokens": 100,
                    "tokens_per_second": throughput, "wall_s": 100 / throughput,
                    "request_metrics": {key: dist for key in bench.previous.REQUEST_METRICS},
                    "batch_event_gap_distribution_s": dist,
                    "per_request_delivery_gap_distribution_s": dist,
                    "peak_used_pages": i + 2, "peak_reserved_kv_slots": (i + 2) * 256,
                    "peak_gpu_allocated_bytes": 100 + i, "peak_gpu_reserved_bytes": 200 + i,
                    "peak_gpu_allocated_delta_bytes": 10 + i, "peak_gpu_reserved_delta_bytes": 20 + i,
                    "mean_effective_output_block_tokens_per_packed_verify_call": 6 + i,
                    "mean_effective_output_block_tokens_per_verified_request": 3 + i,
                    "after_cleanup": {"used_blocks": 0, "free_blocks": pages},
                    "pool": {"shape": [2, 36, pages, 256, 8, 128]}})
            return rows
        upstream = samples("upstream_flashattn", (100, 300, 200), 100)
        jet = samples("jetspec", (450, 600, 500), 80)
        case = bench.combined_summary(upstream, jet)["cases"][0]
        self.assertEqual(case["jetspec_over_upstream_median_throughput_ratio"], 2.5)
        self.assertNotIn("paired_repeat_throughput_ratios", case)
        self.assertEqual([v["mode"] for v in case["variants"]], ["upstream_flashattn", "jetspec"])
        self.assertEqual(case["variants"][0]["max_observed_per_request_delivery_gap_s"], 4)
        self.assertTrue(all(v["all_allocator_cleanup_passed"] for v in case["variants"]))
        self.assertEqual(case["variants"][1]["raw_throughputs_tokens_per_second"], [450, 600, 500])
        with self.assertRaisesRegex(AssertionError, "at least three"):
            bench.combined_summary(upstream[:2], jet)
        changed = deepcopy(jet)
        changed[0]["actual_output_tokens"] = 101
        with self.assertRaisesRegex(AssertionError, "unequal real token counts"):
            bench.combined_summary(upstream, changed)


if __name__ == "__main__":
    unittest.main()
