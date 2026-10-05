#!/usr/bin/env python3
"""Phase-5 reference/candidate JetSpec serving on the frozen final manifest.

Each --mode worker loads an independent clean source snapshot. It uses the
same serving policy, delivery ledger, wall-clock arrivals, cold allocator,
and latency formulas as the final matched benchmark. No attention function
is monkeypatched: the production dispatch of each snapshot is measured.
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

import jetspec_flash_serving_benchmark as shared

previous, provenance = shared.previous, shared.upstream
require, save = shared.require, shared.save


class TreePathEvidence(AbstractContextManager):
    """Original Python wrapper execution/callers during untimed warmup only."""
    def __init__(self, repo, required_symbol):
        self.root = Path(repo).resolve() / "nanovllm" / "speculative" / "jetspec"
        self.required_symbol = required_symbol
        self.counts = Counter()

    def _profile(self, frame, event, arg):
        if event != "call" or not frame.f_code.co_name.startswith("packed_tree_attention"):
            return
        file = Path(frame.f_code.co_filename).resolve()
        if file.is_relative_to(self.root):
            caller = frame.f_back
            self.counts[(str(file), frame.f_code.co_name,
                         caller.f_code.co_name if caller is not None else None)] += 1

    def __enter__(self):
        require(sys.getprofile() is None, "another profiler would confound evidence")
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)

    def result(self):
        require(any(function == self.required_symbol and count > 0
                    for (_, function, _), count in self.counts.items()),
                f"warmup did not execute expected original wrapper: {self.required_symbol}")
        return {"required_original_symbol": self.required_symbol,
            "calls": [{"file": file, "file_sha256": provenance.file_sha(file),
                       "function": function, "caller": caller, "calls": count}
                      for (file, function, caller), count in sorted(self.counts.items())],
            "method": "untimed first warmup original function sys.setprofile; no replacement",
            "profiler_removed_for_timed_samples": sys.getprofile() is None}


def identity(nano, official):
    source = provenance.source_identity(nano, official)
    source["phase5_harness_sha256"] = provenance.file_sha(__file__)
    source["matched_delivery_helper_sha256"] = provenance.file_sha(previous.__file__)
    source["serving_policy_helper_sha256"] = provenance.file_sha(shared.__file__)
    return source


def worker(args):
    require(args.warmup >= 1 and args.repeats >= 3, "needs >=1 warmup and >=3 repeats per case")
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch, nanovllm, jetspec
    require(torch.cuda.is_available(), "CUDA is required; no CPU fallback")
    require(Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()), "wrong production import")
    initial = identity(nanovllm, jetspec)
    shared.check_candidate_source(initial, args)
    manifest = provenance.load_manifest(args.manifest)
    selected = [case for case in manifest["cases"] if not args.cases or
                f"c{case['concurrency']}_o{case['output_cap_scale']}" in args.cases.split(",")]
    require(selected, "requested matrix has no cases")
    report = {"schema_version": 1, "kind": "Phase-5 same-policy JetSpec tree-kernel serving ablation",
        "status": "in_progress", "passed": False, "label": args.label, "mode": "jetspec",
        "source": initial, "environment": provenance.environment(torch),
        "config": dict(shared.USER_CONFIG), "serving_policy": {**shared.SERVING_POLICY, "attention_backend": "sdpa"},
        "models": {"target": previous._helpers.checkpoint(args.target), "draft": previous._helpers.checkpoint(args.draft)},
        "manifest_path": str(Path(args.manifest).resolve()), "manifest_file_sha256": provenance.file_sha(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "cases": [{k: v for k, v in case.items() if k != "specs"} for case in selected],
        "invocation": vars(args), "measurement": {"warmups_per_case": args.warmup, "repeats": args.repeats,
            "offered_clock": "original final manifest wall-clock arrivals; no retiming",
            "sample_order": "all selected warmups, then case/repeat order; all raw samples retained",
            "delivery": "original exactly-once public token events",
            "representative_case_preregistered": "c8_o512", "target_features_tree_budget_unchanged": True,
            "no_serving_parameter_tuning": True, "enable_chunked_prefill": False,
            "peak_memory_scope": "PyTorch allocated/reserved high-water; not NVML whole-device"},
        "warmups": [], "samples": []}
    save(args.output, report)
    engine = None
    try:
        torch.manual_seed(0)
        engine = nanovllm.LLM(args.target, **shared.USER_CONFIG)
        require({str(p.dtype) for p in engine.model_runner.model.parameters()} == {"torch.bfloat16"}, "Target is not BF16")
        runtime = engine.get_jetspec_batch_runtime(args.draft)
        require(runtime.target is engine.model_runner.model and runtime.kv_pool is engine.model_runner.kv_cache,
                "runtime does not use runner-owned Target/pool")
        report["pool"] = previous.pool_identity(engine.model_runner.kv_cache)
        report["pool"]["block_size"] = engine.scheduler.block_manager.block_size
        report["draft_residency"] = previous.draft_identity(runtime)
        jobs = [("warmup", case, i) for case in selected for i in range(args.warmup)]
        jobs += [("timed", case, i) for case in selected for i in range(args.repeats)]
        for index, (phase, case, repeat) in enumerate(jobs):
            require(identity(nanovllm, jetspec) == initial, "production source changed before sample")
            cold = shared.prepare_candidate(engine, runtime, args.draft, case["concurrency"], "sdpa")
            require(sys.getprofile() is None, "profiler leaked into benchmark")
            print(f"[{index+1}/{len(jobs)}] {args.label} {phase} c{case['concurrency']} O{case['output_cap_scale']} repeat={repeat+1}", flush=True)
            if index == 0:
                with TreePathEvidence(args.repo, args.expected_tree_function) as evidence:
                    result = previous.serve(engine, runtime, case, "jetspec", args.deadline)
                report["actual_tree_path"] = evidence.result()
            else:
                result = previous.serve(engine, runtime, case, "jetspec", args.deadline)
            require(identity(nanovllm, jetspec) == initial, "production source changed during sample")
            require(sys.getprofile() is None, "untimed profiler still active")
            result.update(phase=phase, mode="jetspec", label=args.label, repeat=repeat,
                case_index=case["case_index"], concurrency=case["concurrency"], output_cap_scale=case["output_cap_scale"],
                workload_sha256=case["workload_sha256"], cold_allocator=cold, pool=report["pool"],
                profiler_active_during_timed=False if phase == "timed" else None)
            report["warmups" if phase == "warmup" else "samples"].append(result)
            save(args.output, report)
            print(f"DONE tokens={result['actual_output_tokens']} tok/s={result['tokens_per_second']:.3f}", flush=True)
        report["source_end"] = identity(nanovllm, jetspec)
        require(report["source_end"] == initial, "source changed at completion")
        require(provenance.file_sha(args.manifest) == report["manifest_file_sha256"], "manifest changed")
        require(len(report["samples"]) == len(selected) * args.repeats, "sample matrix incomplete")
        report.update(status="complete", passed=True)
        save(args.output, report)
    except BaseException as error:
        report.update(status="failed", passed=False, failure={"type": type(error).__name__, "message": str(error)})
        save(args.output, report)
        raise
    finally:
        if engine is not None:
            atexit.unregister(engine.exit)
            engine.exit()


def paired_summary(baseline, candidate):
    for worker in (baseline, candidate):
        require(worker["passed"] and worker["status"] == "complete", "incomplete worker")
        require(worker["source"] == worker["source_end"], "source changed")
        require(worker["actual_tree_path"]["profiler_removed_for_timed_samples"], "path profiler leaked")
        require(all(sample["profiler_active_during_timed"] is False for sample in worker["samples"]), "profiled formal sample")
    for key in ("manifest_sha256", "manifest_file_sha256", "cases", "config", "serving_policy", "models"):
        require(baseline[key] == candidate[key], f"worker mismatch: {key}")
    for key in ("torch", "cuda", "gpu", "compute_capability", "executable", "packages"):
        require(baseline["environment"][key] == candidate["environment"][key], f"environment mismatch: {key}")
    result = []
    for case in baseline["cases"]:
        arms = []
        for worker in (baseline, candidate):
            samples = [s for s in worker["samples"] if s["case_index"] == case["case_index"]]
            require(len(samples) >= 3, "fewer than three retained raw samples")
            require(all(s["workload_sha256"] == case["workload_sha256"] for s in samples), "sample workload changed")
            require(all(s["after_cleanup"]["used_blocks"] == 0 and s["exactly_once_and_cap_passed"] for s in samples), "lifetime/delivery failure")
            arms.append({"label": worker["label"], "raw_throughput_tok_s": [s["tokens_per_second"] for s in samples],
                "median_throughput_tok_s": statistics.median(s["tokens_per_second"] for s in samples),
                "median_request_metric_p50": {key: statistics.median(s["request_metrics"][key]["p50"] for s in samples)
                                              for key in previous.REQUEST_METRICS},
                "median_delivery_gap_p95_s": statistics.median(s["per_request_delivery_gap_distribution_s"]["p95"] for s in samples),
                "worst_delivery_gap_max_s": max(s["per_request_delivery_gap_distribution_s"]["max"] for s in samples),
                "peak_gpu_allocated_bytes": max(s["peak_gpu_allocated_bytes"] for s in samples),
                "peak_gpu_reserved_bytes": max(s["peak_gpu_reserved_bytes"] for s in samples),
                "peak_leased_pages": max(s["peak_used_pages"] for s in samples),
                "median_emitted_tokens_per_packed_verify": statistics.median(s["mean_effective_output_block_tokens_per_packed_verify_call"] for s in samples),
                "median_emitted_tokens_per_verified_request": statistics.median(s["mean_effective_output_block_tokens_per_verified_request"] for s in samples),
                "allocator_cleanup_all_passed": True})
        result.append({**case, "arms": arms, "median_candidate_over_baseline_throughput":
                       arms[1]["median_throughput_tok_s"] / arms[0]["median_throughput_tok_s"]})
    return result


def combine(args):
    baseline = json.loads(Path(args.baseline_results).read_text())
    candidate = json.loads(Path(args.candidate_results).read_text())
    summary = paired_summary(baseline, candidate)
    save(args.output, {"status": "complete", "passed": True,
        "kind": "Phase-5 same-policy tree kernel serving ablation; not upstream AR comparison",
        "summary": summary, "baseline_worker": baseline, "candidate_worker": candidate,
        "raw_artifacts": [{"path": str(Path(path).resolve()), "sha256": provenance.file_sha(path)}
                          for path in (args.baseline_results, args.candidate_results)],
        "postprocessing_harness_sha256": provenance.file_sha(__file__),
        "numerical_contract_note": "Serving token hashes are evidence, not cross-layout bitwise equivalence requirements; separate unchanged numerical and isolation qualification is mandatory."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker", "combine"), required=True)
    for flag in ("repo", "target", "draft", "manifest", "expected-head", "expected-production-sha",
                 "baseline-results", "candidate-results"):
        parser.add_argument("--" + flag)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-tree-function", default="packed_tree_attention")
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--deadline", type=float, default=1800)
    parser.add_argument("--cases", default="", help="subset e.g c1_o128,c8_o512; original case specs never changed")
    args = parser.parse_args()
    if args.mode == "worker":
        require(all(getattr(args, field) for field in ("repo", "target", "draft", "manifest", "expected_head", "expected_production_sha")),
                "worker requires source/model/manifest pins")
        worker(args)
    else:
        require(args.baseline_results and args.candidate_results, "combine requires both complete workers")
        combine(args)


if __name__ == "__main__":
    main()
