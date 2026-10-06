"""CPU-only gates for immutable workloads and three-arm publication."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import jetspec_official_port_benchmark as bench


def fake_worker(mode, label, values=(100, 300, 200), *, graph=False):
    cases = [{"case_index": i, "concurrency": c, "output_cap_scale": o,
              "workload_sha256": f"workload-{i}"}
             for i, (c, o) in enumerate((c, o) for c in (1, 4, 8) for o in (128, 512))]
    source = {"production_sha256": label, "git": {"status_porcelain": "",
              "head": bench.upstream.UPSTREAM_HEAD if mode == "upstream_flashattn" else label}}
    metrics = {key: {"p50": .2, "p95": .4, "max": .5} for key in bench.previous.REQUEST_METRICS}
    def sample(case, value, repeat):
        return {**case, "repeat": repeat, "tokens_per_second": value,
            "actual_output_tokens": 100, "wall_s": 100 / value,
            "profiler_active_during_timed": False, "after_cleanup": {"used_blocks": 0, "free_blocks": 100},
            "exactly_once_and_cap_passed": True, "pool": {"shape": [2, 36, 100, 256, 8, 128]},
            "request_metrics": deepcopy(metrics),
            "per_request_delivery_gap_distribution_s": {"p95": .3, "max": .6},
            "peak_gpu_allocated_bytes": 1000, "peak_gpu_reserved_bytes": 2000,
            "peak_used_pages": 20, "mean_effective_output_block_tokens_per_packed_verify_call": 30 if mode == "jetspec" else None,
            "mean_effective_output_block_tokens_per_verified_request": 6 if mode == "jetspec" else None,
            "target_graph_delta": {"captures": 0, "replays": 5, "eager_fallbacks": 0, "staged_bytes": 100} if graph else None}
    evidence = {"profiler_removed_for_timed_samples": True}
    execution = "cuda_graph" if graph else "eager"
    if mode == "upstream_flashattn":
        evidence.update(sdpa_calls=0, counts={api: 10 for api in
            ("flash_attn_varlen_func", "flash_attn_with_kvcache")})
    else:
        evidence.update(execution=execution, kernels="reference")
    return {"passed": True, "status": "complete", "label": label, "mode": mode,
        "source": source, "source_end": deepcopy(source), "actual_path": evidence,
        "manifest_sha256": "same", "manifest_file_sha256": "same", "cases": cases,
        "config": dict(bench.policy.USER_CONFIG),
        "serving_policy": {**bench.policy.SERVING_POLICY, "attention_backend": "sdpa"} if mode == "jetspec" else None,
        "execution_policy": {"target_execution": execution, "target_kernels": "reference"} if mode == "jetspec" else
                            {"target_execution": "eager", "attention_backend": "flash_attn"},
        "measurement": {"warmups_per_case": 1, "repeats": len(values)},
        "models": {"target": "same", **({"draft": "same"} if mode == "jetspec" else {})},
        "environment": {**{key: "same" for key in ("torch", "cuda", "gpu", "compute_capability", "executable", "python")},
            "packages": {key: "same" for key in ("torch", "transformers", "triton", "numpy", "flash-attn")}},
        "warmups": [sample(case, values[0], 0) for case in cases],
        "samples": [sample(case, value, repeat) for case in cases for repeat, value in enumerate(values)]}


class OfficialPortBenchmarkCPU(unittest.TestCase):
    def test_three_arm_medians_do_not_cherry_pick_best_run(self):
        upstream = fake_worker("upstream_flashattn", "upstream", (100, 300, 200))
        old = fake_worker("jetspec", "old", (300, 500, 400))
        new = fake_worker("jetspec", "new", (410, 1000, 500), graph=True)
        rows = bench.combined_summary(upstream, new, old)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[5]["jetspec_over_upstream_median_throughput_ratio"], 500 / 200)
        self.assertEqual(rows[5]["optimized_over_old_jetspec_median_throughput_ratio"], 500 / 400)
        self.assertEqual(rows[5]["arms"][2]["raw_throughput_tok_s"], [410, 1000, 500])
        self.assertEqual(rows[5]["arms"][2]["worst_delivery_gap_max_s"], .6)

    def test_same_user_config_and_environment_are_required(self):
        upstream = fake_worker("upstream_flashattn", "upstream")
        for field in ("manifest_sha256", "manifest_file_sha256", "cases", "config"):
            new = fake_worker("jetspec", "new")
            new[field] = "changed"
            with self.assertRaises((AssertionError, TypeError)):
                bench.combined_summary(upstream, new)
        new = fake_worker("jetspec", "new")
        new["environment"]["packages"]["flash-attn"] = "different"
        with self.assertRaisesRegex(AssertionError, "package differs"):
            bench.combined_summary(upstream, new)

    def test_lifetime_sampling_source_and_flash_failures_block_publication(self):
        edits = [lambda w: w["source_end"].update(production_sha256="changed"),
                 lambda w: w["samples"][0].update(profiler_active_during_timed=True),
                 lambda w: w["samples"][0].update(tokens_per_second=123),
                 lambda w: w["samples"][0]["after_cleanup"].update(used_blocks=1),
                 lambda w: w["warmups"].clear(), lambda w: w.update(passed=False)]
        for edit in edits:
            new = fake_worker("jetspec", "new")
            edit(new)
            with self.assertRaises(AssertionError):
                bench.validate_worker(new)
        upstream = fake_worker("upstream_flashattn", "upstream")
        upstream["actual_path"]["sdpa_calls"] = 1
        with self.assertRaisesRegex(AssertionError, "FlashAttention"):
            bench.validate_worker(upstream)

    def test_subset_requires_explicit_diagnostic_publication(self):
        worker = fake_worker("jetspec", "new")
        worker["cases"] = worker["cases"][5:]
        worker["samples"] = worker["samples"][-3:]
        worker["warmups"] = worker["warmups"][-1:]
        with self.assertRaisesRegex(AssertionError, "full six-case"):
            bench.validate_worker(worker)
        bench.validate_worker(worker, allow_subset=True)

    def test_case_selection_never_rebuilds_workload(self):
        manifest = {"cases": fake_worker("jetspec", "new")["cases"]}
        selected = bench.select_cases(manifest, "c1_o128,c8_o512")
        self.assertIs(selected[0], manifest["cases"][0])
        self.assertIs(selected[1], manifest["cases"][5])
        for names in ("c2_o512", "c8_o512,c8_o512", "c1_o128,"):
            with self.assertRaises(AssertionError):
                bench.select_cases(manifest, names)

    def test_actual_path_profiler_is_always_removed(self):
        self.assertIsNone(sys.getprofile())
        with self.assertRaises(RuntimeError):
            with bench.TargetPathEvidence(Path(__file__).parents[1], "eager", "reference", "packed_tree_attention"):
                raise RuntimeError("fake warmup failure")
        self.assertIsNone(sys.getprofile())

    def test_graph_counter_deltas_preserve_formal_cold_captures(self):
        after = dict(captures=3, replays=10, eager_fallbacks=2, staged_bytes=500)
        self.assertEqual(bench.graph_delta(None, after), after)
        before = dict(captures=2, replays=4, eager_fallbacks=1, staged_bytes=200)
        self.assertEqual(bench.graph_delta(before, after), dict(captures=1, replays=6, eager_fallbacks=1, staged_bytes=300))

    def test_declared_raw_repeats_cannot_be_dropped(self):
        worker = fake_worker("jetspec", "new", (1, 2, 3, 1000))
        worker["samples"] = [s for s in worker["samples"] if s["repeat"] != 3]
        with self.assertRaisesRegex(AssertionError, "retain every declared"):
            bench.validate_worker(worker)

    def test_failed_rms_prototype_is_not_a_serving_choice(self):
        self.assertEqual(bench.TARGET_KERNELS, ("reference", "fused_rope"))


if __name__ == "__main__":
    unittest.main()
