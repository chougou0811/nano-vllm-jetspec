"""CPU-only serving comparison/metric evidence gates."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import jetspec_tree_serving_benchmark as bench


def fake_worker(label, values):
    case = {"case_index": 5, "concurrency": 8, "output_cap_scale": 512, "workload_sha256": "same"}
    source = {"production_sha256": label, "git": {"status_porcelain": "", "head": label}}
    samples = []
    for repeat, value in enumerate(values):
        samples.append({**case, "repeat": repeat, "tokens_per_second": value,
            "profiler_active_during_timed": False, "after_cleanup": {"used_blocks": 0},
            "exactly_once_and_cap_passed": True,
            "request_metrics": {key: {"p50": 10 / value} for key in bench.previous.REQUEST_METRICS},
            "per_request_delivery_gap_distribution_s": {"p95": 20 / value, "max": 30 / value},
            "peak_gpu_allocated_bytes": 100 + repeat, "peak_gpu_reserved_bytes": 200 + repeat,
            "peak_used_pages": 40 + repeat,
            "mean_effective_output_block_tokens_per_packed_verify_call": 30 + repeat,
            "mean_effective_output_block_tokens_per_verified_request": 6 + repeat})
    return {"passed": True, "status": "complete", "label": label, "source": source,
        "source_end": deepcopy(source), "actual_tree_path": {"profiler_removed_for_timed_samples": True},
        "manifest_sha256": "same", "manifest_file_sha256": "same", "cases": [case],
        "config": dict(bench.shared.USER_CONFIG), "serving_policy": dict(bench.shared.SERVING_POLICY),
        "models": {"target": "same", "draft": "same"},
        "environment": {key: "same" for key in ("torch", "cuda", "gpu", "compute_capability", "executable", "packages")},
        "samples": samples}


class TreeServingBenchmarkCPU(unittest.TestCase):
    def test_same_policy_different_source_is_allowed_and_outliers_remain(self):
        baseline = fake_worker("reference", (100, 300, 200))
        candidate = fake_worker("optimized", (210, 400, 390))
        result = bench.paired_summary(baseline, candidate)[0]
        self.assertEqual(result["arms"][0]["raw_throughput_tok_s"], [100, 300, 200])
        self.assertEqual(result["arms"][1]["raw_throughput_tok_s"], [210, 400, 390])
        self.assertEqual(result["median_candidate_over_baseline_throughput"], 390 / 200)
        self.assertEqual(result["arms"][0]["median_emitted_tokens_per_packed_verify"], 31)
        self.assertEqual(result["arms"][0]["median_emitted_tokens_per_verified_request"], 7)
        self.assertEqual(result["arms"][0]["worst_delivery_gap_max_s"], .3)

    def test_config_models_arrivals_and_environment_must_match(self):
        baseline = fake_worker("ref", (1, 2, 3))
        for field in ("manifest_sha256", "manifest_file_sha256", "config", "serving_policy", "models", "cases"):
            candidate = fake_worker("new", (2, 3, 4))
            candidate[field] = "different"
            with self.assertRaises(AssertionError):
                bench.paired_summary(baseline, candidate)
        candidate = fake_worker("new", (2, 3, 4))
        candidate["environment"]["torch"] = "different"
        with self.assertRaisesRegex(AssertionError, "environment mismatch"):
            bench.paired_summary(baseline, candidate)

    def test_min_repeats_source_freeze_and_actual_path_gates(self):
        baseline = fake_worker("ref", (1, 2, 3))
        changes = [lambda w: w["samples"].pop(),
            lambda w: w["source_end"].update(production_sha256="changed"),
            lambda w: w["actual_tree_path"].update(profiler_removed_for_timed_samples=False),
            lambda w: w["samples"][0].update(profiler_active_during_timed=True),
            lambda w: w.update(status="failed", passed=False),
            lambda w: w["samples"][0].update(workload_sha256="changed")]
        for change in changes:
            candidate = fake_worker("new", (2, 3, 4))
            change(candidate)
            with self.assertRaises(AssertionError):
                bench.paired_summary(baseline, candidate)

    def test_allocator_and_exactly_once_failures_are_not_performance_results(self):
        baseline = fake_worker("ref", (1, 2, 3))
        for field in ("after_cleanup", "exactly_once_and_cap_passed"):
            candidate = fake_worker("new", (2, 3, 4))
            candidate["samples"][0][field] = {"used_blocks": 1} if field == "after_cleanup" else False
            with self.assertRaisesRegex(AssertionError, "lifetime/delivery failure"):
                bench.paired_summary(baseline, candidate)

    def test_untimed_path_profiler_always_restores_and_requires_actual_symbol(self):
        self.assertIsNone(sys.getprofile())
        with self.assertRaisesRegex(RuntimeError, "fake"):
            with bench.TreePathEvidence(Path(__file__).parents[1], "packed_tree_attention"):
                raise RuntimeError("fake")
        self.assertIsNone(sys.getprofile())
        with bench.TreePathEvidence(Path(__file__).parents[1], "packed_tree_attention") as evidence:
            pass
        with self.assertRaisesRegex(AssertionError, "did not execute"):
            evidence.result()


if __name__ == "__main__":
    unittest.main()
