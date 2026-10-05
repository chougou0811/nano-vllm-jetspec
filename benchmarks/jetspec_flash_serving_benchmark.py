#!/usr/bin/env python3
"""FlashAttention JetSpec serving comparison, without changing the workload.

Three independent workers can be run: pristine upstream, candidate JetSpec
with FlashAttention, and the same candidate with SDPA.  This driver reuses the
established immutable manifest, delivery ledger, native upstream observer and
metric formulas.  It does not replace production attention/sampling functions.
"""
from __future__ import annotations

import argparse
import atexit
from collections import Counter
from contextlib import AbstractContextManager
import json
from pathlib import Path
import statistics
import sys

import jetspec_final_upstream_flashattn as upstream

previous = upstream.previous
require, save, sha = upstream.require, upstream.save, upstream.sha
BACKENDS = ("sdpa", "flash_attn")
SERVING_POLICY = dict(tree_depth=15, tree_width=7, max_tree_budget=63,
    default_tree_budget=63, max_admissions_per_step=2, max_prefill_tokens=4096,
    enable_chunked_prefill=False, optimization="serving")
USER_CONFIG = dict(max_num_seqs=8, max_model_len=4096,
    max_num_batched_tokens=4096, gpu_memory_utilization=.8,
    tensor_parallel_size=1, enforce_eager=True, kvcache_block_size=256)


class CandidateCallEvidence(AbstractContextManager):
    """Trace original wrappers and the tree kernel in an untimed warmup only."""
    def __init__(self, flash, packed_tree_attention, backend):
        require(backend in BACKENDS, "unknown candidate backend")
        self.backend = backend
        self.functions = {name: getattr(flash, name) for name in
            ("flash_attn_varlen_func", "flash_attn_with_kvcache")}
        self.functions["packed_tree_attention"] = packed_tree_attention
        require(all(callable(f) and hasattr(f, "__code__") for f in self.functions.values()),
                "original attention Python functions are unavailable")
        self.codes = {f.__code__: name for name, f in self.functions.items()}
        self.counts, self.callers = Counter(), Counter()
        self.sdpa_calls = 0
        self.old_profile = None

    def _profile(self, frame, event, arg):
        if event == "call" and frame.f_code in self.codes:
            name = self.codes[frame.f_code]
            self.counts[name] += 1
            caller = frame.f_back
            if caller is not None:
                self.callers[(name, caller.f_code.co_filename, caller.f_code.co_name)] += 1
        elif event == "c_call" and getattr(arg, "__name__", "") == "scaled_dot_product_attention":
            self.sdpa_calls += 1

    def __enter__(self):
        self.old_profile = sys.getprofile()
        require(self.old_profile is None, "an existing profiler would confound path qualification")
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(self.old_profile)

    def result(self):
        require(self.counts["packed_tree_attention"] > 0, "warmup missed packed Target verification")
        if self.backend == "flash_attn":
            require(self.counts["flash_attn_varlen_func"] > 0,
                    "requested FlashAttention did not execute the original varlen API")
        else:
            require(not any(self.counts[name] for name in
                ("flash_attn_varlen_func", "flash_attn_with_kvcache")),
                "explicit SDPA candidate unexpectedly executed external FlashAttention")
        return {"phase": "untimed original-manifest warmup", "backend": self.backend,
            "method": "sys.setprofile counts original imported function code objects and callers; no monkeypatch",
            "counts": {name: self.counts[name] for name in self.functions},
            "callers": [{"api": api, "file": path, "function": function, "calls": count}
                for (api, path, function), count in sorted(self.callers.items())],
            "sdpa_calls": self.sdpa_calls,
            "functions": {name: {"module": f.__module__, "file": f.__code__.co_filename,
                "file_sha256": upstream.file_sha(f.__code__.co_filename)}
                for name, f in self.functions.items()},
            "packed_verification_backend": "existing Triton paged tree attention, unchanged",
            "with_kvcache_required_for_candidate": False,
            "profiler_removed_for_timed_samples": sys.getprofile() is None}


def candidate_source_identity(nano, official_jetspec):
    result = upstream.source_identity(nano, official_jetspec)
    result["driver_sha256"] = upstream.file_sha(__file__)
    result["upstream_driver_sha256"] = upstream.file_sha(upstream.__file__)
    return result


def prepare_candidate(engine, runtime, draft, concurrency, backend):
    """Cold metadata, fixed tree policy, and an explicit idle-boundary backend."""
    from nanovllm.engine.block_manager import BlockManager
    previous.cleanup(engine, runtime)
    old = engine.scheduler.block_manager
    manager = BlockManager(len(old.blocks), old.block_size)
    engine.scheduler.block_manager = manager
    runtime.block_manager = manager
    runtime.arena.block_manager = manager
    engine.model_runner.config.max_num_seqs = concurrency
    engine.scheduler.max_num_seqs = concurrency
    require(not manager.hash_to_block_id and all(b.hash == -1 and not b.token_ids for b in manager.blocks),
            "new allocator is not cold")
    serving = engine.configure_jetspec(draft, **SERVING_POLICY, attention_backend=backend)
    require(serving.runtime is runtime and serving.max_num_seqs == concurrency,
            "serving factory changed cached runtime or concurrency")
    require(runtime._lightweight, "timed candidate accidentally enables diagnostic events")
    require(runtime._attention_backend == backend, "requested attention backend was not applied")
    require(runtime._prefill_attention_backend == "sdpa", "unqualified FA Target prefill selected")
    return previous.idle_boundary(engine, runtime, allow_scratch=False)


def check_candidate_source(source, args):
    require(source["git"]["head"].startswith(args.expected_head), "wrong candidate HEAD")
    require(source["production_sha256"] == args.expected_production_sha,
            "candidate production fingerprint differs from the preregistered snapshot")
    require(args.allow_dirty_source or source["git"]["status_porcelain"] == "",
            "candidate snapshot is dirty; use a clean worktree or explicitly record --allow-dirty-source")


def candidate_worker(args):
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch, nanovllm, jetspec, flash_attn, flash_attn_2_cuda
    from nanovllm.speculative.jetspec.paged_backend import packed_tree_attention
    require(torch.cuda.is_available(), "CUDA is required")
    require(Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()),
            "candidate import escaped --repo")
    initial = candidate_source_identity(nanovllm, jetspec)
    check_candidate_source(initial, args)
    manifest = upstream.load_manifest(args.manifest)
    mode = "jetspec_flashattn" if args.backend == "flash_attn" else "jetspec_sdpa"
    report = {"schema_version": 1, "status": "in_progress", "mode": mode,
        "backend": args.backend, "source": initial, "environment": upstream.environment(torch),
        "config": dict(USER_CONFIG), "invocation": {"argv": sys.argv, "arguments": vars(args)},
        "models": {"target": previous._helpers.checkpoint(args.target),
                   "draft": previous._helpers.checkpoint(args.draft)},
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_file_sha256": upstream.file_sha(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "cases": [{k: v for k, v in case.items() if k != "specs"} for case in manifest["cases"]],
        "flash_extension": {"path": str(Path(flash_attn_2_cuda.__file__).resolve()),
                            "sha256": upstream.file_sha(flash_attn_2_cuda.__file__)},
        "policy": {"warmup_per_case": args.warmup, "timed_repeats_per_case": args.repeats,
            "allocator_start": "new BlockManager before every sample; within-sample production prefix policy unchanged",
            "mode_isolation": "independent worker processes; same FlashAttention environment for both candidate flags",
            "timed_order": "all six case warmups, then each case's three timed repeats; all samples retained",
            "sample_clock": "offered arrival wall clock, production stepping, token ledger, final synchronization; setup/cleanup excluded",
            "max_num_seqs": "case concurrency, set on existing config and scheduler",
            "initialization_max_num_seqs": 8,
            "representative_case_preregistered": {"concurrency": 8, "output_cap_scale": 512},
            "greedy_temperature": 0, "ignore_eos": True, "enable_chunked_prefill": False,
            "optimization": "serving", "attention_backend": args.backend,
            "delivery_observation": "existing public exactly-once token events",
            "no_benchmark_parameter_tuning": True,
            "source_clean": initial["git"]["status_porcelain"] == "",
            "gpu_peak_memory": "PyTorch allocated/reserved high-water marks, not NVML total-device peaks"},
        "attention_scope": {"draft": args.backend, "target_prefill": "sdpa",
                            "target_tree_verify": "qualified FP32 TILE64 Triton"},
        "resolved_jetspec_policy": {**SERVING_POLICY, "attention_backend": args.backend,
                                   "max_verify_tokens": 4096, "record_timing": False},
        "warmups": [], "samples": [], "source_frozen_checks": []}
    engine = runtime = None
    save(args.output, report)
    try:
        torch.manual_seed(0)
        engine = nanovllm.LLM(args.target, **USER_CONFIG)
        require({str(p.dtype) for p in engine.model_runner.model.parameters()} == {"torch.bfloat16"},
                "Target parameter dtype is not BF16")
        pool = engine.model_runner.kv_cache
        require(pool.dtype == torch.bfloat16, "KV dtype is not BF16")
        report["pool"] = previous.pool_identity(pool)
        report["pool"]["block_size"] = engine.scheduler.block_manager.block_size
        report["target_parameter_dtypes"] = ["torch.bfloat16"]
        runtime = engine.get_jetspec_batch_runtime(args.draft)
        require(runtime.target is engine.model_runner.model and runtime.kv_pool is pool,
                "candidate escaped its runner Target/KV")
        report["draft_residency"] = previous.draft_identity(runtime)
        jobs = [("warmup", case, i) for case in manifest["cases"] for i in range(args.warmup)]
        jobs += [("timed", case, i) for case in manifest["cases"] for i in range(args.repeats)]
        save(args.output, report)
        for index, (phase, case, repeat) in enumerate(jobs):
            require(candidate_source_identity(nanovllm, jetspec) == initial, "source changed before sample")
            cold = prepare_candidate(engine, runtime, args.draft, case["concurrency"], args.backend)
            require(all(report["pool"][key] == value for key, value in previous.pool_identity(pool).items()),
                    "KV pool changed")
            require(sys.getprofile() is None, "a profiler is active before sample")
            print(f"[{index + 1}/{len(jobs)}] {mode} {phase} c{case['concurrency']} O{case['output_cap_scale']} repeat={repeat + 1} START", flush=True)
            if phase == "warmup" and index == 0:
                with CandidateCallEvidence(flash_attn, packed_tree_attention, args.backend) as evidence:
                    sample = previous.serve(engine, runtime, case, "jetspec", args.deadline)
                report["attention_actual_path"] = evidence.result()
            else:
                sample = previous.serve(engine, runtime, case, "jetspec", args.deadline)
            require(sys.getprofile() is None, "qualification profiler leaked into timed path")
            sample.update(phase=phase, mode=mode, repeat=repeat, case_index=case["case_index"],
                concurrency=case["concurrency"], output_cap_scale=case["output_cap_scale"],
                workload_sha256=case["workload_sha256"], cold_allocator=cold, pool=report["pool"],
                profiler_active_during_timed=False if phase == "timed" else None,
                attention_backend=args.backend)
            require(candidate_source_identity(nanovllm, jetspec) == initial, "source changed during sample")
            report["source_frozen_checks"].append({"job": index, "before_after_equal": True})
            report["warmups" if phase == "warmup" else "samples"].append(sample)
            save(args.output, report)
            print(f"[{index + 1}/{len(jobs)}] DONE tokens={sample['actual_output_tokens']} wall={sample['wall_s']:.3f}s tok/s={sample['tokens_per_second']:.3f} pages={sample['peak_used_pages']}", flush=True)
        require(len(report["warmups"]) == 6 * args.warmup and len(report["samples"]) == 6 * args.repeats,
                "candidate sample matrix incomplete")
        report["source_end"] = candidate_source_identity(nanovllm, jetspec)
        require(report["source_end"] == initial, "candidate changed at completion")
        require(upstream.file_sha(args.manifest) == report["manifest_file_sha256"], "manifest changed")
        report.update(status="complete", passed=True)
        save(args.output, report)
    except BaseException as error:
        report.update(status="failed", passed=False,
                      failure={"type": type(error).__name__, "message": str(error)})
        save(args.output, report)
        raise
    finally:
        if engine is not None:
            atexit.unregister(engine.exit)
            engine.exit()


def validate_candidate_evidence(worker):
    expected_mode = "jetspec_flashattn" if worker["backend"] == "flash_attn" else "jetspec_sdpa"
    require(worker["mode"] == expected_mode, "candidate mode/backend disagree")
    evidence = worker["attention_actual_path"]
    require(evidence["backend"] == worker["backend"], "path evidence/backend disagree")
    require(evidence["profiler_removed_for_timed_samples"], "candidate qualification profiler was not removed")
    require(evidence["counts"]["packed_tree_attention"] > 0, "missing packed Target verification evidence")
    if worker["backend"] == "flash_attn":
        require(evidence["counts"]["flash_attn_varlen_func"] > 0, "missing actual candidate FlashAttention calls")
        actual_callers = {(Path(row["file"]).name, row["function"])
                         for row in evidence["callers"]
                         if row["api"] == "flash_attn_varlen_func" and row["calls"] > 0}
        require(("flash_draft.py", "_flash_group") in actual_callers,
                "candidate must execute real FlashAttention in Draft")
        require(("flash_prefill.py", "flash_causal_prefill") not in actual_callers and
                evidence["sdpa_calls"] > 0,
                "serving Target prefill must retain its qualified SDPA path")
    else:
        require(evidence["counts"]["flash_attn_varlen_func"] == 0 and
                evidence["counts"]["flash_attn_with_kvcache"] == 0, "SDPA ablation executed external FlashAttention")
    require(worker["policy"]["enable_chunked_prefill"] is False, "chunking unexpectedly enabled")
    require(worker["source_end"] == worker["source"], "candidate source changed")


def validate_worker_pair(baseline, candidate, *, candidate_ablation=False):
    for worker in (baseline, candidate):
        require(worker["passed"] and worker["status"] == "complete", "refusing an incomplete worker")
        require(len(worker["warmups"]) >= 6 and len(worker["samples"]) >= 18,
                "worker did not run at least one warmup and three repeats per case")
        require(all(s.get("profiler_active_during_timed") is False for s in worker["samples"]),
                "timed sample had a qualification profiler")
    for key in ("manifest_sha256", "manifest_file_sha256", "config", "cases"):
        require(baseline[key] == candidate[key], f"workers do not match {key}")
    require(baseline["models"]["target"] == candidate["models"]["target"], "checkpoint/tokenizer differs")
    for key in ("torch", "cuda", "gpu", "compute_capability", "executable", "packages"):
        require(baseline["environment"][key] == candidate["environment"][key], f"worker environment differs: {key}")
    validate_candidate_evidence(candidate)
    if candidate_ablation:
        validate_candidate_evidence(baseline)
        require(baseline["backend"] == "sdpa" and candidate["backend"] == "flash_attn", "ablation flags swapped")
        require(baseline["source"] == candidate["source"], "ablation does not use the same candidate source")
        require(baseline["flash_extension"] == candidate["flash_extension"], "ablation Flash extension differs")
        require(baseline["resolved_jetspec_policy"] ==
                {**candidate["resolved_jetspec_policy"], "attention_backend": "sdpa"},
                "ablation changed a serving parameter other than backend")
    else:
        require(baseline["mode"] == "upstream_flashattn", "baseline is not native upstream")
        require(baseline["source"]["git"]["head"] == upstream.UPSTREAM_HEAD and
                baseline["source"]["git"]["status_porcelain"] == "", "upstream snapshot is not pristine")
        require(baseline["flash_attention_actual_path"]["sdpa_calls"] == 0 and
                baseline["flash_attention_actual_path"]["profiler_removed_for_timed_samples"],
                "upstream FlashAttention path proof failed")


def comparison_summary(baseline_samples, candidate_samples, *, ablation=False):
    result = upstream.combined_summary(baseline_samples, candidate_samples)
    baseline_mode = "jetspec_sdpa" if ablation else "upstream_flashattn"
    for case in result["cases"]:
        ratio = case.pop("jetspec_over_upstream_median_throughput_ratio")
        case["flashattn_over_sdpa_median_throughput_ratio" if ablation else
             "jetspec_flashattn_over_upstream_median_throughput_ratio"] = ratio
        case["variants"][0]["mode"] = baseline_mode
        case["variants"][1]["mode"] = "jetspec_flashattn"
        # The reused two-arm helper maps arm 0 to ordinary AR and therefore
        # intentionally drops its emitted-block statistics. In this experiment
        # arm 0 may be JetSpec SDPA. Restore both arms from their unchanged raw
        # samples, not from the helper's role-based ordinary-AR placeholder.
        for variant, samples in zip(case["variants"], (baseline_samples, candidate_samples)):
            runs = [sample for sample in samples if
                    (sample["concurrency"], sample["output_cap_scale"], sample["workload_sha256"]) ==
                    (case["concurrency"], case["output_cap_scale"], case["workload_sha256"])]
            for summary_key, sample_key in (
                ("median_effective_output_block_tokens_per_packed_verify_call",
                 "mean_effective_output_block_tokens_per_packed_verify_call"),
                ("median_effective_output_block_tokens_per_verified_request",
                 "mean_effective_output_block_tokens_per_verified_request")):
                values = [sample[sample_key] for sample in runs if sample[sample_key] is not None]
                variant[summary_key] = statistics.median(values) if values else None
    if ablation:
        result["comparison"] = "same candidate source/environment/user and serving parameters; only attention_backend differs"
    return result


def explanatory_profile(path, candidate):
    evidence = json.loads(Path(path).read_text())
    require(evidence.get("passed") is True and evidence.get("status") == "complete" and
            evidence.get("NOT_A_SERVING_BENCHMARK") is True,
            "explanatory Draft profile is incomplete or not explicitly separated from serving data")
    require(evidence["source"]["production_sha256"] == candidate["source"]["production_sha256"] and
            evidence["source"]["source_file_sha256"] == candidate["source"]["source_file_sha256"] and
            evidence["source"]["git"]["head"] == candidate["source"]["git"]["head"],
            "explanatory Draft profile used different measured production")
    require(evidence["models"] == candidate["models"], "explanatory profile checkpoints differ")
    require(all(evidence["environment"][key] == candidate["environment"][key]
                for key in ("torch", "cuda", "gpu", "compute_capability", "executable")),
            "explanatory profile environment differs")
    return {"path": str(Path(path).resolve()), "sha256": upstream.file_sha(path),
            "NOT_SERVING_DATA": True, "must_not_use_for_serving_speedup": True,
            "measured_production_models_environment_bound": True,
            "interpretation": "Same-input Draft-only microtiming and untimed profiler; explains attention/packing costs, not aggregate serving throughput or its speedup.",
            "evidence": evidence}


def bind_correctness_source(correctness, candidate):
    """Bind an untimed qualification to exactly the measured production bytes.

    The two legacy fingerprint helpers include different benchmark helpers and
    use different key names. Compare the complete production file maps instead
    of incorrectly comparing their differently scoped aggregate hashes.
    """
    require(correctness.get("passed") is True and correctness.get("status") == "complete",
            "candidate correctness qualification did not pass")
    for name in ("all_gates_passed", "source_unchanged", "harness_unchanged",
                 "allocator_clean", "tree_contract_unchanged"):
        require(correctness.get(name) is True, f"qualification proof missing: {name}")
    source = correctness["source"]
    mapped = {}
    for name, digest in source["production_file_sha256"].items():
        if name.startswith("nanovllm/"):
            mapped["nanovllm:" + name[len("nanovllm/"):]] = digest
        elif name.startswith("official_jetspec:"):
            mapped["jetspec:" + name[len("official_jetspec:"):]] = digest
    require(mapped == candidate["source"]["source_file_sha256"],
            "qualification and benchmark loaded different production bytes")
    require(source["revision"] == candidate["source"]["git"]["head"] and
            source["worktree_status"] == "", "qualification is not the measured clean revision")
    require(source["tree_backend"]["unchanged_from_frozen"], "qualification tree contract changed")
    paths = source["fingerprint_scope"]["model_paths"]
    require(all(str(Path(paths[name]).resolve()) == candidate["models"][name]["path"]
                for name in ("target", "draft")), "qualification checkpoint paths differ")
    require(all(correctness["environment"][name] == candidate["environment"][name]
                for name in ("torch", "cuda", "gpu")), "qualification environment differs")
    return {"all_production_files_equal": True, "production_file_count": len(mapped),
            "clean_revision_equal": True, "checkpoint_paths_equal": True,
            "torch_cuda_gpu_equal": True, "tree_backend_unchanged": True}


def combine(args):
    native = json.loads(Path(args.upstream_results).read_text())
    candidate = json.loads(Path(args.jetspec_results).read_text())
    validate_worker_pair(native, candidate)
    correctness = json.loads(Path(args.correctness_results).read_text())
    require(correctness.get("passed") is True, "candidate correctness qualification did not pass")
    correctness_binding = bind_correctness_source(correctness, candidate)
    certificate = json.loads(Path(args.flash_validation).read_text())
    require(certificate["status"] == "passed" and certificate["varlen_callable"] and
            certificate["kvcache_callable"] and all(c["gpu_call_passed"] for c in certificate["cases"]),
            "independent FlashAttention GPU certificate did not pass")
    require({c["api"] for c in certificate["cases"]} ==
            {"flash_attn_varlen_func", "flash_attn_with_kvcache"}, "GPU certificate missed a Flash API")
    require((certificate["torch"], certificate["cuda"], certificate["gpu"]) ==
            tuple(candidate["environment"][key] for key in ("torch", "cuda", "gpu")),
            "Flash GPU certificate used a different device/software")
    summary = comparison_summary(native["samples"], candidate["samples"])
    require(len(summary["cases"]) == 6, "comparison matrix incomplete")
    representative = next(c for c in summary["cases"] if
        (c["concurrency"], c["output_cap_scale"]) == (8, 512))
    report = {"schema_version": 1, "status": "complete", "passed": True,
        "benchmark_type": "pristine upstream FlashAttention vs Draft-FlashAttention JetSpec practical system comparison",
        "baseline_revision_selection": {"preferred_commit": "bb823b3e06983d71485a8e1f23715ebd87d98ef8",
            "selected_commit": upstream.UPSTREAM_HEAD,
            "reason": "preferred project-start commit rejects greedy; nearest native-greedy official ancestor used without patches"},
        "representative_case_preregistered": {"concurrency": 8, "output_cap_scale": 512},
        "representative_median_speedup": representative[
            "jetspec_flashattn_over_upstream_median_throughput_ratio"],
        "summary": summary, "upstream_worker": native, "jetspec_flashattn_worker": candidate,
        "postprocessing_source": {"driver_path": str(Path(__file__).resolve()),
            "driver_sha256": upstream.file_sha(__file__),
            "upstream_metric_helper_sha256": upstream.file_sha(upstream.__file__),
            "matched_metric_helper_sha256": upstream.file_sha(previous.__file__),
            "worker_source_fingerprints_preserved": True,
            "note": "These are publisher/postprocessing bytes, which may differ from each worker's recorded timed harness. Original worker source/driver fingerprints remain unmodified."},
        "correctness_qualification": {"path": str(Path(args.correctness_results).resolve()),
            "sha256": upstream.file_sha(args.correctness_results), "evidence": correctness,
            "measured_production_binding": correctness_binding},
        "flash_attention_gpu_validation": {"path": str(Path(args.flash_validation).resolve()),
            "sha256": upstream.file_sha(args.flash_validation), "evidence": certificate},
        "artifacts": [{"mode": worker["mode"], "path": str(Path(path).resolve()), "sha256": upstream.file_sha(path)}
            for worker, path in ((native, args.upstream_results), (candidate, args.jetspec_results))],
        "interpretation": "Same original mixed-cap two-wave user workload, not uniform output lengths. Different backends/layouts need not be token-bitwise equal; exactly-once caps/lifetime and the existing numerical contract must pass. FA is used in Draft only: Target prefill retains SDPA after its separate FA negative qualification; packed Target tree verification remains the validated FP32 Triton backend. Speedups measure the whole serving system; no algorithm-only attribution is claimed."}
    if args.sdpa_results:
        sdpa = json.loads(Path(args.sdpa_results).read_text())
        validate_worker_pair(sdpa, candidate, candidate_ablation=True)
        report["jetspec_sdpa_worker"] = sdpa
        report["ablation_summary"] = comparison_summary(sdpa["samples"], candidate["samples"], ablation=True)
        report["artifacts"].append({"mode": sdpa["mode"], "path": str(Path(args.sdpa_results).resolve()),
                                    "sha256": upstream.file_sha(args.sdpa_results)})
    if args.profile:
        report["explanatory_draft_profile"] = explanatory_profile(args.profile, candidate)
    save(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("upstream", "jetspec", "combine"), required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="flash_attn")
    for option in ("repo", "target", "draft", "manifest", "upstream-results", "jetspec-results",
                   "sdpa-results", "flash-validation", "correctness-results", "expected-head",
                   "expected-production-sha", "profile"):
        parser.add_argument("--" + option)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--deadline", type=float, default=1800)
    args = parser.parse_args()
    if args.mode == "combine":
        require(args.upstream_results and args.jetspec_results and args.flash_validation and args.correctness_results,
                "combine requires both complete workers, Flash GPU certificate and candidate correctness qualification")
        combine(args)
    else:
        require(args.repo and args.target and args.manifest and (args.mode == "upstream" or args.draft),
                "worker requires repo/checkpoints/immutable manifest")
        require(args.warmup >= 1 and args.repeats >= 3 and args.deadline > 0, "need warmup >=1, repeats >=3")
        if args.mode == "upstream":
            upstream.worker(args)
        else:
            require(args.expected_head and args.expected_production_sha,
                    "candidate requires explicit production HEAD and source fingerprint")
            candidate_worker(args)


if __name__ == "__main__":
    main()
