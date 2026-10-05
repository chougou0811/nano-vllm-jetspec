"""CPU-only gates for the FlashAttention serving experiment and its metrics."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import jetspec_flash_serving_benchmark as bench


def fake_functions():
    def varlen():
        return 11
    def kvcache():
        return 12
    def packed():
        return 13
    return SimpleNamespace(flash_attn_varlen_func=varlen,
                           flash_attn_with_kvcache=kvcache), packed


def metric_samples(mode, throughputs=(100, 300, 200), pages=91):
    samples = []
    for index, value in enumerate(throughputs):
        distribution = bench.previous.distribution((index + 1, index + 2))
        samples.append({"mode": mode, "concurrency": 8, "output_cap_scale": 512,
            "repeat": index, "workload_sha256": "identical", "actual_output_tokens": 100,
            "tokens_per_second": value, "wall_s": 100 / value,
            "request_metrics": {key: distribution for key in bench.previous.REQUEST_METRICS},
            "batch_event_gap_distribution_s": distribution,
            "per_request_delivery_gap_distribution_s": distribution,
            "peak_used_pages": index + 2, "peak_reserved_kv_slots": (index + 2) * 256,
            "peak_gpu_allocated_bytes": 100 + index, "peak_gpu_reserved_bytes": 200 + index,
            "peak_gpu_allocated_delta_bytes": 10 + index, "peak_gpu_reserved_delta_bytes": 20 + index,
            "mean_effective_output_block_tokens_per_packed_verify_call": 6 + index,
            "mean_effective_output_block_tokens_per_verified_request": 3 + index,
            "after_cleanup": {"used_blocks": 0, "free_blocks": pages},
            "pool": {"shape": [2, 36, pages, 256, 8, 128]},
            "profiler_active_during_timed": False})
    return samples


def worker(backend="flash_attn"):
    mode = "jetspec_flashattn" if backend == "flash_attn" else "jetspec_sdpa"
    source = {"production_sha256": "frozen", "git": {"head": "abcdef", "status_porcelain": ""}}
    return {"mode": mode, "backend": backend, "passed": True, "status": "complete",
        "source": source, "source_end": deepcopy(source), "manifest_sha256": "same",
        "manifest_file_sha256": "same-file", "config": dict(bench.USER_CONFIG),
        "cases": list(range(6)), "models": {"target": "same-checkpoint"},
        "environment": {"torch": "2.8", "cuda": "12.8", "gpu": "5090",
            "compute_capability": [12, 0], "executable": "/same/python", "packages": {"flash-attn": "2.8.3"}},
        "warmups": [None] * 6, "samples": metric_samples(mode) * 6,
        "policy": {"enable_chunked_prefill": False},
        "resolved_jetspec_policy": {**bench.SERVING_POLICY, "attention_backend": backend},
        "flash_extension": {"sha256": "same-extension"},
        "attention_actual_path": {"backend": backend, "profiler_removed_for_timed_samples": True,
            "sdpa_calls": 36,
            "callers": [{"api": "flash_attn_varlen_func", "file": "/repo/flash_draft.py",
                         "function": "_flash_group", "calls": 54}],
            "counts": {"packed_tree_attention": 36, "flash_attn_varlen_func": 90 if backend == "flash_attn" else 0,
                       "flash_attn_with_kvcache": 0}}}


class FlashServingBenchmarkTests(unittest.TestCase):
    def test_flash_warmup_counts_varlen_and_unchanged_tree_without_requiring_kvcache(self):
        flash, packed = fake_functions()
        with bench.CandidateCallEvidence(flash, packed, "flash_attn") as evidence:
            flash.flash_attn_varlen_func()
            flash.flash_attn_varlen_func()
            packed()
        result = evidence.result()
        self.assertEqual(result["counts"], {"flash_attn_varlen_func": 2,
            "flash_attn_with_kvcache": 0, "packed_tree_attention": 1})
        self.assertIsNone(sys.getprofile())
        self.assertTrue(result["profiler_removed_for_timed_samples"])
        self.assertEqual({row["api"] for row in result["callers"]},
                         {"flash_attn_varlen_func", "packed_tree_attention"})

    def test_sdpa_warmup_rejects_external_flash_calls(self):
        flash, packed = fake_functions()
        with bench.CandidateCallEvidence(flash, packed, "sdpa") as evidence:
            packed()
        self.assertEqual(evidence.result()["counts"]["flash_attn_varlen_func"], 0)
        with bench.CandidateCallEvidence(flash, packed, "sdpa") as evidence:
            packed()
            flash.flash_attn_varlen_func()
        with self.assertRaisesRegex(AssertionError, "unexpectedly executed"):
            evidence.result()

    def test_missing_requested_backend_or_packed_path_fails(self):
        flash, packed = fake_functions()
        with bench.CandidateCallEvidence(flash, packed, "flash_attn") as evidence:
            packed()
        with self.assertRaisesRegex(AssertionError, "did not execute"):
            evidence.result()
        with bench.CandidateCallEvidence(flash, packed, "flash_attn") as evidence:
            flash.flash_attn_varlen_func()
        with self.assertRaisesRegex(AssertionError, "missed packed"):
            evidence.result()

    def test_call_profiler_is_removed_even_when_warmup_raises(self):
        flash, packed = fake_functions()
        with self.assertRaisesRegex(RuntimeError, "fake failure"):
            with bench.CandidateCallEvidence(flash, packed, "flash_attn"):
                raise RuntimeError("fake failure")
        self.assertIsNone(sys.getprofile())

    def test_explicit_source_fingerprint_and_dirty_disclosure_are_required(self):
        source = worker()["source"]
        args = SimpleNamespace(expected_head="abc", expected_production_sha="frozen", allow_dirty_source=False)
        bench.check_candidate_source(source, args)
        wrong = deepcopy(source)
        wrong["production_sha256"] = "changed"
        with self.assertRaisesRegex(AssertionError, "fingerprint differs"):
            bench.check_candidate_source(wrong, args)
        dirty = deepcopy(source)
        dirty["git"]["status_porcelain"] = " M nanovllm/config.py"
        with self.assertRaisesRegex(AssertionError, "snapshot is dirty"):
            bench.check_candidate_source(dirty, args)
        args.allow_dirty_source = True
        bench.check_candidate_source(dirty, args)

    def test_prepare_keeps_old_workload_policy_and_applies_new_explicit_backend(self):
        class Manager:
            def __init__(self, count, size):
                self.blocks = [SimpleNamespace(hash=-1, token_ids=[]) for _ in range(count)]
                self.block_size = size
                self.hash_to_block_id = {}
        runtime = SimpleNamespace(arena=SimpleNamespace(block_manager=None), _lightweight=True,
                                  _prefill_attention_backend="sdpa")
        engine = SimpleNamespace(scheduler=SimpleNamespace(block_manager=Manager(91, 256)),
                                 model_runner=SimpleNamespace(config=SimpleNamespace()))
        captured = {}
        def configure(draft, **kwargs):
            captured.update(kwargs)
            captured["draft"] = draft
            runtime._attention_backend = kwargs["attention_backend"]
            return SimpleNamespace(runtime=runtime, max_num_seqs=engine.model_runner.config.max_num_seqs)
        engine.configure_jetspec = configure
        with patch.dict(sys.modules, {"nanovllm.engine.block_manager": SimpleNamespace(BlockManager=Manager)}), \
             patch.object(bench.previous, "cleanup"), \
             patch.object(bench.previous, "idle_boundary", return_value={"free_blocks": 91}):
            result = bench.prepare_candidate(engine, runtime, "draft-checkpoint", 4, "flash_attn")
        self.assertEqual(result, {"free_blocks": 91})
        self.assertEqual(captured, {**bench.SERVING_POLICY, "draft": "draft-checkpoint", "attention_backend": "flash_attn"})
        self.assertEqual(engine.scheduler.max_num_seqs, 4)
        self.assertEqual(len(engine.scheduler.block_manager.blocks), 91)
        self.assertIs(runtime.block_manager, runtime.arena.block_manager)

    def test_same_source_same_environment_backend_ablation_gate(self):
        sdpa, flash = worker("sdpa"), worker()
        bench.validate_worker_pair(sdpa, flash, candidate_ablation=True)
        broken = deepcopy(sdpa)
        broken["source"]["production_sha256"] = "other"
        broken["source_end"] = deepcopy(broken["source"])
        with self.assertRaisesRegex(AssertionError, "same candidate source"):
            bench.validate_worker_pair(broken, flash, candidate_ablation=True)
        broken = deepcopy(sdpa)
        broken["resolved_jetspec_policy"]["max_tree_budget"] = 127
        with self.assertRaisesRegex(AssertionError, "serving parameter"):
            bench.validate_worker_pair(broken, flash, candidate_ablation=True)

    def test_different_environment_or_manifest_or_timed_profiler_fails(self):
        sdpa, flash = worker("sdpa"), worker()
        for key, value, message in (("manifest_sha256", "changed", "manifest_sha256"),
                                    ("config", {"max_num_seqs": 9}, "config")):
            broken = deepcopy(flash)
            broken[key] = value
            with self.assertRaisesRegex(AssertionError, message):
                bench.validate_worker_pair(sdpa, broken, candidate_ablation=True)
        broken = deepcopy(flash)
        broken["environment"]["executable"] = "/different/python"
        with self.assertRaisesRegex(AssertionError, "executable"):
            bench.validate_worker_pair(sdpa, broken, candidate_ablation=True)
        broken = deepcopy(flash)
        broken["samples"][0]["profiler_active_during_timed"] = True
        with self.assertRaisesRegex(AssertionError, "qualification profiler"):
            bench.validate_worker_pair(sdpa, broken, candidate_ablation=True)

    def test_pristine_upstream_gate_rejects_patched_or_wrong_baseline(self):
        native = worker("sdpa")
        native["mode"] = "upstream_flashattn"
        native["source"]["git"] = {"head": bench.upstream.UPSTREAM_HEAD, "status_porcelain": ""}
        native["flash_attention_actual_path"] = {"sdpa_calls": 0, "profiler_removed_for_timed_samples": True}
        flash = worker()
        bench.validate_worker_pair(native, flash)
        native["source"]["git"]["status_porcelain"] = " M nanovllm/layers/sampler.py"
        with self.assertRaisesRegex(AssertionError, "not pristine"):
            bench.validate_worker_pair(native, flash)

    def test_comparison_uses_three_run_medians_and_preserves_negative_results(self):
        native = metric_samples("upstream_flashattn", (100, 300, 200))
        flash = metric_samples("jetspec_flashattn", (450, 600, 500))
        case = bench.comparison_summary(native, flash)["cases"][0]
        self.assertEqual(case["jetspec_flashattn_over_upstream_median_throughput_ratio"], 2.5)
        self.assertEqual([v["mode"] for v in case["variants"]], ["upstream_flashattn", "jetspec_flashattn"])
        self.assertNotIn("paired_repeat_throughput_ratios", case)
        self.assertEqual(case["variants"][1]["raw_throughputs_tokens_per_second"], [450, 600, 500])
        faster_sdpa = metric_samples("jetspec_sdpa", (800, 900, 1000))
        ablation = bench.comparison_summary(faster_sdpa, flash, ablation=True)["cases"][0]
        self.assertAlmostEqual(ablation["flashattn_over_sdpa_median_throughput_ratio"], 500 / 900)
        self.assertLess(ablation["flashattn_over_sdpa_median_throughput_ratio"], 1)
        self.assertEqual(ablation["variants"][0]["max_observed_per_request_delivery_gap_s"], 4)

    def test_flash_serving_proof_rejects_experimental_target_prefill(self):
        flash = worker()
        flash["attention_actual_path"]["callers"].append({
            "api": "flash_attn_varlen_func", "file": "/repo/flash_prefill.py",
            "function": "flash_causal_prefill", "calls": 36})
        with self.assertRaisesRegex(AssertionError, "qualified SDPA"):
            bench.validate_candidate_evidence(flash)

    def test_ablation_restores_both_emitted_medians_from_unchanged_raw_samples(self):
        sdpa = metric_samples("jetspec_sdpa")
        flash = metric_samples("jetspec_flashattn")
        for row, packed, per_request in zip(sdpa, (11, 13, 12), (3, 7, 5)):
            row["mean_effective_output_block_tokens_per_packed_verify_call"] = packed
            row["mean_effective_output_block_tokens_per_verified_request"] = per_request
        for row, packed, per_request in zip(flash, (15, 14, 19), (8, 6, 7)):
            row["mean_effective_output_block_tokens_per_packed_verify_call"] = packed
            row["mean_effective_output_block_tokens_per_verified_request"] = per_request
        before = deepcopy((sdpa, flash))
        variants = bench.comparison_summary(sdpa, flash, ablation=True)["cases"][0]["variants"]
        self.assertEqual([row["median_effective_output_block_tokens_per_packed_verify_call"]
                          for row in variants], [12, 15])
        self.assertEqual([row["median_effective_output_block_tokens_per_verified_request"]
                          for row in variants], [5, 7])
        self.assertEqual((sdpa, flash), before)

    def test_native_nonverification_stats_stay_none_and_zero_is_not_dropped(self):
        native = metric_samples("upstream_flashattn")
        flash = metric_samples("jetspec_flashattn")
        for row in native:
            row["mean_effective_output_block_tokens_per_packed_verify_call"] = None
            row["mean_effective_output_block_tokens_per_verified_request"] = None
        for row in flash:
            row["mean_effective_output_block_tokens_per_packed_verify_call"] = 0
            row["mean_effective_output_block_tokens_per_verified_request"] = 0
        variants = bench.comparison_summary(native, flash)["cases"][0]["variants"]
        for name in ("median_effective_output_block_tokens_per_packed_verify_call",
                     "median_effective_output_block_tokens_per_verified_request"):
            self.assertIsNone(variants[0][name])
            self.assertEqual(variants[1][name], 0)

    def test_optional_profile_is_bound_and_labelled_nonserving_without_mutating_workers(self):
        flash = worker()
        flash["source"]["source_file_sha256"] = {"nanovllm:model.py": "bytes"}
        evidence = {"passed": True, "status": "complete", "NOT_A_SERVING_BENCHMARK": True,
            "source": deepcopy(flash["source"]), "environment": deepcopy(flash["environment"]),
            "models": deepcopy(flash["models"]), "microtimings": [{"cpu_wall_ms": 10}]}
        original = deepcopy(flash)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(evidence))
            attached = bench.explanatory_profile(path, flash)
            self.assertEqual(attached["evidence"], evidence)
            self.assertTrue(attached["NOT_SERVING_DATA"])
            self.assertTrue(attached["must_not_use_for_serving_speedup"])
            self.assertEqual(attached["sha256"], bench.upstream.file_sha(path))
            self.assertEqual(flash, original)
            changed = deepcopy(evidence)
            changed["source"]["production_sha256"] = "other"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(AssertionError, "different measured production"):
                bench.explanatory_profile(path, flash)
            changed = deepcopy(evidence)
            changed["NOT_A_SERVING_BENCHMARK"] = False
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(AssertionError, "separated from serving"):
                bench.explanatory_profile(path, flash)

    def test_qualification_binding_uses_complete_production_bytes_not_helper_hash(self):
        flash = worker()
        flash["models"] = {"target": {"path": "/checkpoints/target"},
                           "draft": {"path": "/checkpoints/draft"}}
        flash["source"]["source_file_sha256"] = {
            "nanovllm:model.py": "target-byte-hash", "jetspec:head.py": "draft-byte-hash"}
        qualified = {"passed": True, "status": "complete", "all_gates_passed": True,
            "source_unchanged": True, "harness_unchanged": True, "allocator_clean": True,
            "tree_contract_unchanged": True, "environment": flash["environment"],
            "source": {"revision": "abcdef", "worktree_status": "",
                "production_file_sha256": {"nanovllm/model.py": "target-byte-hash",
                    "official_jetspec:head.py": "draft-byte-hash",
                    "loaded_benchmark_helper:jetspec_phase3.py": "different-helper-scope"},
                "tree_backend": {"unchanged_from_frozen": True},
                "fingerprint_scope": {"model_paths": {"target": "/checkpoints/target",
                                                       "draft": "/checkpoints/draft"}}}}
        self.assertEqual(bench.bind_correctness_source(qualified, flash)["production_file_count"], 2)
        broken = deepcopy(qualified)
        broken["source"]["production_file_sha256"]["official_jetspec:head.py"] = "modified"
        with self.assertRaisesRegex(AssertionError, "production bytes"):
            bench.bind_correctness_source(broken, flash)
        broken = deepcopy(qualified)
        broken["allocator_clean"] = False
        with self.assertRaisesRegex(AssertionError, "allocator_clean"):
            bench.bind_correctness_source(broken, flash)


if __name__ == "__main__":
    unittest.main()
