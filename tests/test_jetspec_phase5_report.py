"""CPU-only final publication gates; fixtures are synthetic, not benchmark data."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "benchmarks" / "jetspec_phase5_report.py"
spec = importlib.util.spec_from_file_location("phase5_report", path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def source(label):
    return {"source_file_sha256": {"nanovllm:kernel.py": label, "jetspec:draft.py": "official"},
            "git": {"head": label, "status_porcelain": ""}}


def fake_worker(label, gain=1):
    cases = [{"case_index": index, "concurrency": c, "output_cap_scale": o,
              "workload_sha256": f"manifest-{index}"}
             for index, (c, o) in enumerate(report.EXPECTED_MATRIX)]
    samples, warmups = [], []
    for case in cases:
        for repeat, speed in enumerate((100, 300, 200)):
            speed *= gain
            samples.append({**case, "repeat": repeat, "tokens_per_second": speed,
                "actual_output_tokens": 1000, "wall_s": 1000 / speed,
                "profiler_active_during_timed": False, "exactly_once_and_cap_passed": True,
                "after_cleanup": {"used_blocks": 0}, "requests": [{"token_ids": [1, 2, 3]}],
                "request_metrics": {key: {"p50": 100 / speed} for key in report.REQUEST_METRICS},
                "per_request_delivery_gap_distribution_s": {"p95": 100 / speed, "max": 200 / speed},
                "peak_gpu_allocated_bytes": 100, "peak_gpu_reserved_bytes": 200, "peak_used_pages": 10,
                "mean_effective_output_block_tokens_per_packed_verify_call": 32,
                "mean_effective_output_block_tokens_per_verified_request": 7})
        warmups.append(deepcopy(samples[-3]))
    src = source(label)
    return {"status": "complete", "passed": True, "label": label, "source": src, "source_end": deepcopy(src),
        "manifest_sha256": "manifest", "manifest_file_sha256": "manifest-file", "cases": cases,
        "config": {"tensor_parallel_size": 1, "enforce_eager": True, "max_model_len": 4096},
        "serving_policy": {"enable_chunked_prefill": False, "attention_backend": "sdpa", "tree_budget": 63},
        "models": {"target": {"path": "/target"}, "draft": {"path": "/draft"}},
        "environment": {key: "same" for key in
            ("torch", "cuda", "gpu", "compute_capability", "executable", "packages")},
        "actual_tree_path": {"profiler_removed_for_timed_samples": True,
            "required_original_symbol": "packed_tree_attention", "calls": [{"function": "packed_tree_attention", "calls": 36}]},
        "samples": samples, "warmups": warmups}


def fake_qualification(candidate):
    src = {"production_file_sha256": {"nanovllm/kernel.py": candidate["label"], "official_jetspec:draft.py": "official"},
        "revision": candidate["label"], "worktree_status": "",
        "fingerprint_scope": {"model_paths": {"target": "/target", "draft": "/draft"}}}
    return {"status": "complete", "passed": True, "all_gates_passed": True, "source_unchanged": True,
        "allocator_cleanup": True, "source": src, "environment": deepcopy(candidate["environment"]),
        "fp32_pre_bf16_scaled_max_and_relative_rms_bound": 1e-4,
        "full_network_bf16_scaled_max_and_relative_rms_bound": 2 ** -6,
        "cross_shape_token_bitwise_required": False, "synthetic": [{"passed": True}],
        "trained_same_state": [{"rounds": [{"passed": True}]}],
        "candidate": {"entry": "module:operator", "module_sha256": "module-sha",
            "function_source_sha256": "function-sha", "executed_options": {"num_warps": 4}},
        "packed_request_isolation": {"real_eight_request_packed_shape": True, "finite_same_shape_controls_passed": True},
        "lifecycle": {"semantic_gates": {"eos": True, "cancel": True}, "fixed_schedule_replay_exact": True},
        "allocator_pressure": {"deferred_seen": True, "recovered": True},
        "preemption_recompute": {key: True for key in ("preemption_seen", "resume_seen", "output_exactly_once",
                                                     "fixed_schedule_replay_exact", "scratch_one_page_or_less")},
        "chunked_lifecycle_recompute": {"output_exactly_once": True}}


def fake_micro(candidate):
    return {"status": "complete", "passed": True, "NOT_A_SERVING_BENCHMARK": True, "source_unchanged": True,
        "source": {"production_files": {"nanovllm/kernel.py": candidate["label"]},
                   "head": candidate["label"], "status_porcelain": ""},
        "candidate": {"module": "module", "function": "operator", "file_sha256": "module-sha", "function_sha256": "function-sha"},
        "candidate_variant": {"num_warps": 4}, "environment": deepcopy(candidate["environment"]),
        "measurement": {"warmups_per_arm": 1, "repeats": 3},
        "cases": [{"validation": {"passed": True}, "samples": {arm: [1, 2, 3] for arm in ("baseline", "candidate")},
                   "warmups": {arm: [1] for arm in ("baseline", "candidate")}}]}


class Phase5ReportCPU(unittest.TestCase):
    def setUp(self):
        self.reference, self.candidate = fake_worker("reference"), fake_worker("candidate", gain=1.5)
        self.micro = fake_micro(self.candidate)
        self.qual = fake_qualification(self.candidate)

    def publish(self, **kwargs):
        return report.publication(self.reference, self.candidate, self.micro, self.qual, **kwargs)

    def test_all_raw_repeats_and_negative_failures_preserved(self):
        failure = {"status": "failed", "passed": False, "error": "fixed numerical gate failed", "raw": [100, 20, 30]}
        result = self.publish(negatives=[("dot", failure, {"path": "/failed.json", "sha256": "raw-sha"})])
        self.assertTrue(result["passed"])
        self.assertEqual(result["representative_median_serving_speedup"], 1.5)
        self.assertEqual(result["serving_summary"][-1]["arms"][0]["raw_throughput_tok_s"], [100, 300, 200])
        self.assertEqual(result["negative_results"][0]["evidence"], failure)
        self.assertEqual(len(result["reference_worker"]["samples"]), 18)
        self.assertEqual(len(result["reference_worker"]["warmups"]), 6)
        self.assertNotIn("requests", result["reference_worker"]["samples"][0])
        self.assertIn("requests", self.reference["samples"][0])

    def test_failed_or_synthetic_only_candidate_cannot_be_published(self):
        for change in (lambda q: q.update(passed=False, status="failed"),
                       lambda q: q.update(trained_same_state=[]),
                       lambda q: q.update(allocator_cleanup=False),
                       lambda q: q.update(fp32_pre_bf16_scaled_max_and_relative_rms_bound=1e-3),
                       lambda q: q.update(full_network_bf16_scaled_max_and_relative_rms_bound=.03)):
            self.qual = fake_qualification(self.candidate)
            change(self.qual)
            with self.assertRaises(ValueError):
                self.publish()

    def test_lifecycle_isolation_recompute_must_be_real_passes(self):
        changes = [lambda q: q["packed_request_isolation"].update(real_eight_request_packed_shape=False),
            lambda q: q["lifecycle"]["semantic_gates"].update(eos=False),
            lambda q: q["allocator_pressure"].update(recovered=False),
            lambda q: q["preemption_recompute"].update(resume_seen=False),
            lambda q: q["chunked_lifecycle_recompute"].update(output_exactly_once=False)]
        for change in changes:
            self.qual = fake_qualification(self.candidate)
            change(self.qual)
            with self.assertRaises(ValueError):
                self.publish()

    def test_pre_dispatch_source_cannot_be_claimed_as_final_qualification(self):
        self.qual["source"]["production_file_sha256"]["nanovllm/kernel.py"] = "old"
        with self.assertRaisesRegex(ValueError, "production files differ"):
            self.publish()

    def test_micro_source_function_options_and_environment_must_bind(self):
        changes = [lambda m: m["source"]["production_files"].update({"nanovllm/kernel.py": "old"}),
            lambda m: m["candidate"].update(function_sha256="old"),
            lambda m: m.update(candidate_variant={}),
            lambda m: m["environment"].update(torch="other"),
            lambda m: m.update(NOT_A_SERVING_BENCHMARK=False),
            lambda m: m["cases"][0]["samples"].update(candidate=[1, 2])]
        for change in changes:
            self.micro = fake_micro(self.candidate)
            change(self.micro)
            with self.assertRaises(ValueError):
                self.publish()

    def test_practical_matching_does_not_compare_internal_source_but_requires_policy(self):
        self.assertTrue(self.publish()["passed"])
        for key in ("manifest_sha256", "config", "serving_policy", "models"):
            self.candidate = fake_worker("candidate", gain=1.5)
            self.candidate[key] = "other"
            with self.assertRaises((ValueError, AttributeError)):
                self.publish()

    def test_fewer_repeats_duplicate_ids_missing_warmups_and_edited_workload_rejected(self):
        changes = [lambda w: w["samples"].pop(), lambda w: w["samples"][0].update(repeat=1),
            lambda w: w.update(warmups=[]), lambda w: w["samples"][0].update(workload_sha256="other"),
            lambda w: w["samples"][0].update(profiler_active_during_timed=True),
            lambda w: w["samples"][0].update(tokens_per_second=999),
            lambda w: w["samples"][0]["after_cleanup"].update(used_blocks=1)]
        for change in changes:
            self.candidate = fake_worker("candidate", gain=1.5)
            change(self.candidate)
            with self.assertRaises(ValueError):
                self.publish()

    def test_profile_is_diagnostic_bound_and_never_supplies_claimed_speedup(self):
        profile = deepcopy(self.candidate)
        profile.update(diagnostic_only=True, policy=deepcopy(self.candidate["serving_policy"]),
                       stage_run={"tokens_per_second": 99999, "requests": []})
        result = self.publish(profiles=[("candidate", profile, {"path": "/profile", "sha256": "profile-sha"})])
        self.assertEqual(result["representative_median_serving_speedup"], 1.5)
        self.assertTrue(result["diagnostic_profiles"][0]["evidence"]["must_not_use_for_serving_throughput"])
        profile["source"] = source("other")
        with self.assertRaises(ValueError):
            self.publish(profiles=[("candidate", profile, {})])

    def test_named_artifacts_are_explicit_and_unique(self):
        self.assertEqual(report.named_paths(["failed=/x.json", "slow=/y.json"]), [("failed", "/x.json"), ("slow", "/y.json")])
        for inputs in (["missing"], ["=/x"], ["a="], ["a=/x", "a=/y"]):
            with self.assertRaises(ValueError):
                report.named_paths(inputs)


if __name__ == "__main__":
    unittest.main()
