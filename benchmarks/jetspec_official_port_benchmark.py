#!/usr/bin/env python3
"""Official-inspired Target optimization against pristine FlashAttention nano-vLLM.

Independent, clean, source-pinned workers reuse the frozen final workload and
exactly-once delivery ledger. No production wrapper is replaced. CUDA Graph
replay is an explicit JetSpec system difference, not a backend-matched ablation.
All warmups and raw formal samples are retained; c8/output512 is preregistered.
"""
from __future__ import annotations

import argparse
import atexit
from collections import Counter
from contextlib import AbstractContextManager
import inspect
import json
from pathlib import Path
import statistics
import sys

import jetspec_final_upstream_flashattn as upstream
import jetspec_flash_serving_benchmark as policy

previous = upstream.previous
require, save, sha = upstream.require, upstream.save, upstream.sha
TARGET_EXECUTIONS = ("eager", "cuda_graph")
TARGET_KERNELS = ("reference", "fused_rope")
CASE_NAMES = tuple(f"c{c}_o{o}" for c in (1, 4, 8) for o in (128, 512))


def select_cases(manifest, selection=""):
    names = selection.split(",") if selection else list(CASE_NAMES)
    require(len(set(names)) == len(names) and all(name in CASE_NAMES for name in names),
            "unknown or duplicate benchmark case")
    return [case for case in manifest["cases"] if
            f"c{case['concurrency']}_o{case['output_cap_scale']}" in names]


def source_identity(nano, official=None):
    source = upstream.source_identity(nano, official)
    source["driver_sha256"] = upstream.file_sha(__file__)
    source["harness_file_sha256"] = {Path(module.__file__).name: upstream.file_sha(module.__file__)
        for module in (upstream, policy, previous)}
    return source


def check_source(source, expected_head, expected_production_sha, *, native=False):
    require(source["git"]["head"].startswith(expected_head), "wrong production HEAD")
    require(source["git"]["status_porcelain"] == "", "production snapshot is not clean")
    require(source["production_sha256"] == expected_production_sha, "wrong production source fingerprint")
    if native:
        require(source["git"]["head"] == upstream.UPSTREAM_HEAD,
                "native benchmark requires the pristine df99418 greedy-capable ancestor")
    elif source.get("official_jetspec_git") is not None:
        require(source["official_jetspec_git"]["status_porcelain"] == "", "official Draft source is dirty")


class TargetPathEvidence(AbstractContextManager):
    """Trace original function entry/callers only in the first untimed warmup."""
    def __init__(self, repo, execution, kernels, required_tree_function):
        self.root = Path(repo).resolve() / "nanovllm" / "speculative" / "jetspec"
        self.execution, self.kernels = execution, kernels
        self.required_tree_function = required_tree_function
        self.counts = Counter()

    def _profile(self, frame, event, arg):
        if event != "call":
            return
        file, function = Path(frame.f_code.co_filename).resolve(), frame.f_code.co_name
        if not file.is_relative_to(self.root):
            return
        if (function.startswith("packed_tree_attention") or
                file.name in ("target_graph.py", "tree_fusion.py", "tree_norm.py")):
            caller = frame.f_back
            self.counts[(str(file), function, caller.f_code.co_name if caller else None)] += 1

    def __enter__(self):
        require(sys.getprofile() is None, "another profiler would confound actual-path evidence")
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)

    def result(self):
        def called(file_name, function):
            return any(Path(file).name == file_name and name == function and n > 0
                       for (file, name, _), n in self.counts.items())
        require(any(name == self.required_tree_function and n > 0
                    for (_, name, _), n in self.counts.items()), "warmup missed original packed attention wrapper")
        graph_called = called("target_graph.py", "verify")
        require(graph_called == (self.execution == "cuda_graph"), "requested Target execution was not observed")
        rope_called = called("tree_fusion.py", "rope_scatter_prevalidated")
        norm_called = (called("tree_norm.py", "reference_rms_norm") or
                       called("tree_norm.py", "rms_norm"))
        if self.kernels == "fused_rope":
            require(rope_called, "warmup missed original fused RoPE/scatter wrapper")
        require(not norm_called, "unqualified RMS prototype unexpectedly entered the serving path")
        if self.kernels == "reference":
            require(not rope_called, "reference Target unexpectedly used fused RoPE/scatter")
        return {"phase": "first untimed original-manifest warmup", "execution": self.execution,
            "kernels": self.kernels, "required_original_tree_symbol": self.required_tree_function,
            "method": "sys.setprofile original function entries/callers; no replacement",
            "calls": [{"file": file, "file_sha256": upstream.file_sha(file),
                       "function": name, "caller": caller, "count": count}
                      for (file, name, caller), count in sorted(self.counts.items())],
            "graph_verify_observed": graph_called, "fused_rope_observed": rope_called,
            "fused_norm_observed": norm_called,
            "rms_prototype_policy": "negative numerical result; not a supported production serving option",
            "profiler_removed_for_timed_samples": sys.getprofile() is None}


def graph_snapshot(runtime):
    graph = getattr(runtime, "_target_graph", None)
    return graph.snapshot() if graph is not None else None


def graph_delta(before, after):
    if after is None:
        return None
    return {key: after[key] - (before or {}).get(key, 0)
            for key in ("captures", "replays", "eager_fallbacks", "staged_bytes")}


def prepare_jetspec(engine, runtime, args, concurrency):
    """Cold page allocator, retained immutable weights and bounded graph cache."""
    from nanovllm.engine.block_manager import BlockManager
    before_graph = getattr(runtime, "_target_graph", None)
    previous.cleanup(engine, runtime)
    require(getattr(runtime, "_target_graph", None) is before_graph, "cleanup unexpectedly dropped warm graphs")
    old = engine.scheduler.block_manager
    manager = BlockManager(len(old.blocks), old.block_size)
    engine.scheduler.block_manager = manager
    runtime.block_manager = runtime.arena.block_manager = manager
    engine.model_runner.config.max_num_seqs = engine.scheduler.max_num_seqs = concurrency
    require(not manager.hash_to_block_id and all(b.hash == -1 and not b.token_ids for b in manager.blocks),
            "new allocator is not cold")
    parameters = inspect.signature(engine.configure_jetspec).parameters
    keywords = dict(policy.SERVING_POLICY)
    if "attention_backend" in parameters:
        keywords["attention_backend"] = "sdpa"
    else:
        # Frozen Phase 4 b388330 predates this optional policy API and only
        # exposes its original SDPA implementation, with no backend fields.
        require(getattr(runtime, "_attention_backend", "sdpa") == "sdpa" and
                getattr(runtime, "_prefill_attention_backend", "sdpa") == "sdpa",
                "snapshot without backend API does not use the original SDPA policy")
    # The old frozen JetSpec worker legitimately has no graph/kernel policy API.
    for name, value, default in (("target_execution", args.target_execution, "eager"),
                                 ("target_kernels", args.target_kernels, "reference")):
        if name in parameters:
            keywords[name] = value
        else:
            require(value == default, f"snapshot has no {name} API for requested optimization")
    serving = engine.configure_jetspec(args.draft, **keywords)
    require(serving.runtime is runtime and serving.max_num_seqs == concurrency, "serving runtime/config changed")
    require(runtime._lightweight and getattr(runtime, "_attention_backend", "sdpa") == "sdpa" and
            getattr(runtime, "_prefill_attention_backend", "sdpa") == "sdpa",
            "unqualified or diagnostic attention policy selected")
    require(getattr(runtime, "_target_execution", "eager") == args.target_execution, "Target execution not applied")
    require(getattr(runtime, "_target_kernels", "reference") == args.target_kernels, "Target kernels not applied")
    return previous.idle_boundary(engine, runtime, allow_scratch=False)


def worker(args):
    require(args.warmup >= 1 and args.repeats >= 3 and args.deadline > 0, "need warmup >=1 and formal repeats >=3")
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch, nanovllm
    from nanovllm.layers import attention
    official = None
    if args.mode == "jetspec":
        import jetspec as official
    require(torch.cuda.is_available(), "CUDA required; no CPU or SDPA baseline fallback")
    require(Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()), "production import escaped repo")
    initial = source_identity(nanovllm, official)
    check_source(initial, args.expected_head, args.expected_production_sha, native=args.mode == "upstream")
    manifest = upstream.load_manifest(args.manifest)
    selected = select_cases(manifest, args.cases)
    mode = "upstream_flashattn" if args.mode == "upstream" else "jetspec"
    execution = {"target_execution": args.target_execution, "target_kernels": args.target_kernels}
    report = {"schema_version": 1, "kind": "official-inspired Target serving optimization",
        "status": "in_progress", "passed": False, "mode": mode, "label": args.label,
        "source": initial, "environment": upstream.environment(torch), "config": dict(policy.USER_CONFIG),
        "models": {"target": previous._helpers.checkpoint(args.target)},
        "manifest_path": str(Path(args.manifest).resolve()), "manifest_file_sha256": upstream.file_sha(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"], "cases": [{k: v for k, v in c.items() if k != "specs"} for c in selected],
        "serving_policy": {**policy.SERVING_POLICY, "attention_backend": "sdpa"} if official else None,
        "execution_policy": execution if official else {"target_execution": "eager", "attention_backend": "flash_attn"},
        "invocation": {"argv": sys.argv, "arguments": vars(args)},
        "measurement": {"warmups_per_case": args.warmup, "repeats": args.repeats,
            "case_subset_is_diagnostic": len(selected) != 6,
            "offered_clock": "unchanged original two-wave token manifest; user arrival wall clock",
            "sample_order": "each case: clear graph cache outside timer, warmup, then >=3 formal repeats; no samples discarded",
            "allocator_start": "fresh native BlockManager every sample; weights/graph cache stay resident",
            "graph_cache": "independent case cache; resident within case across cold page-allocator resets; all formal captures/fallbacks included",
            "representative_case_preregistered": {"concurrency": 8, "output_cap_scale": 512},
            "greedy_temperature": 0, "ignore_eos": True, "enable_chunked_prefill": False,
            "profiling": "first untimed warmup only; no diagnostic CUDA events in formal samples",
            "peak_memory_scope": "PyTorch allocated/reserved high-water, not NVML whole-device",
            "no_workload_or_serving_parameter_tuning": True},
        "case_graph_resets": [], "warmups": [], "samples": []}
    save(args.output, report)
    engine = runtime = None
    try:
        if args.mode == "upstream":
            import flash_attn, flash_attn_2_cuda
            require(attention.flash_attn_varlen_func is flash_attn.flash_attn_varlen_func and
                    attention.flash_attn_with_kvcache is flash_attn.flash_attn_with_kvcache,
                    "native attention does not use original FlashAttention bindings")
            require("scaled_dot_product_attention" not in Path(attention.__file__).read_text(), "baseline SDPA fallback detected")
            require(nanovllm.SamplingParams(temperature=0).temperature == 0, "native upstream greedy unavailable")
            report["flash_extension"] = {"path": str(Path(flash_attn_2_cuda.__file__).resolve()),
                "sha256": upstream.file_sha(flash_attn_2_cuda.__file__)}
            report["flash_apis_available"] = {name: callable(getattr(attention, name)) for name in
                ("flash_attn_varlen_func", "flash_attn_with_kvcache")}
        torch.manual_seed(0)
        engine = nanovllm.LLM(args.target, **policy.USER_CONFIG)
        require({p.dtype for p in engine.model_runner.model.parameters()} == {torch.bfloat16}, "Target is not BF16")
        pool = engine.model_runner.kv_cache
        require(pool.dtype == torch.bfloat16, "KV pool is not BF16")
        report["pool"] = {**previous.pool_identity(pool), "block_size": engine.scheduler.block_manager.block_size}
        if official:
            runtime = engine.get_jetspec_batch_runtime(args.draft)
            require(runtime.target is engine.model_runner.model and runtime.kv_pool is pool, "runner Target/KV ownership changed")
            report["models"]["draft"] = previous._helpers.checkpoint(args.draft)
            report["draft_residency"] = previous.draft_identity(runtime)
        else:
            report["draft_residency"] = {"loaded": False}
        jobs = [job for c in selected for job in
                [*(("warmup", c, i) for i in range(args.warmup)),
                 *(("timed", c, i) for i in range(args.repeats))]]
        for index, (phase, case, repeat) in enumerate(jobs):
            require(source_identity(nanovllm, official) == initial, "production source changed before sample")
            if runtime is not None and phase == "warmup" and repeat == 0:
                # Independent case qualification: an earlier c1 shape must not
                # exhaust the bounded 16-entry cache before c8's own warmup.
                # This resets only this optional execution cache, outside the
                # timer, retaining the model, Draft weights and fixed policy.
                graph = getattr(runtime, "_target_graph", None)
                old_graph = graph_snapshot(runtime)
                if graph is not None:
                    graph.close()
                    runtime._target_graph = None
                report["case_graph_resets"].append({"case_index": case["case_index"],
                    "scope": "outside timer, before case warmup; model/Draft weights remain resident",
                    "previous_cache": old_graph, "new_cache": None})
            cold = (upstream.prepare_upstream(engine, case["concurrency"]) if runtime is None else
                    prepare_jetspec(engine, runtime, args, case["concurrency"]))
            require(all(report["pool"][key] == value for key, value in previous.pool_identity(pool).items()), "live KV pool changed")
            require(sys.getprofile() is None, "profiler active before sample")
            before_graph = graph_snapshot(runtime) if runtime else None
            print(f"[{index+1}/{len(jobs)}] {args.label} {phase} c{case['concurrency']} O{case['output_cap_scale']} repeat={repeat+1} START", flush=True)
            def serve():
                return (upstream.serve_upstream(engine, case, args.deadline) if runtime is None else
                        previous.serve(engine, runtime, case, "jetspec", args.deadline))
            if index == 0:
                evidence = (upstream.FlashCallEvidence(attention) if runtime is None else
                            TargetPathEvidence(args.repo, args.target_execution, args.target_kernels, args.expected_tree_function))
                with evidence:
                    sample = serve()
                report["actual_path"] = evidence.result()
            else:
                sample = serve()
            after_graph = graph_snapshot(runtime) if runtime else None
            require(sys.getprofile() is None, "untimed profiler leaked into formal samples")
            require(source_identity(nanovllm, official) == initial, "production source changed during sample")
            sample.update(phase=phase, mode=mode, label=args.label, repeat=repeat, case_index=case["case_index"],
                concurrency=case["concurrency"], output_cap_scale=case["output_cap_scale"], workload_sha256=case["workload_sha256"],
                cold_allocator=cold, pool=report["pool"], profiler_active_during_timed=False if phase == "timed" else None,
                target_graph_before=before_graph, target_graph_after=after_graph,
                target_graph_delta=graph_delta(before_graph, after_graph))
            report["warmups" if phase == "warmup" else "samples"].append(sample)
            save(args.output, report)
            print(f"DONE tokens={sample['actual_output_tokens']} tok/s={sample['tokens_per_second']:.3f} graph_delta={sample['target_graph_delta']}", flush=True)
        report["source_end"] = source_identity(nanovllm, official)
        require(report["source_end"] == initial, "source changed at completion")
        require(upstream.file_sha(args.manifest) == report["manifest_file_sha256"], "manifest changed")
        require(len(report["samples"]) == len(selected) * args.repeats, "sample matrix incomplete")
        deltas = [s["target_graph_delta"] for s in report["samples"] if s["target_graph_delta"] is not None]
        captures = sum(d["captures"] for d in deltas)
        report["formal_graph_accounting"] = {"formal_captures": captures,
            "formal_replays": sum(d["replays"] for d in deltas), "formal_eager_fallbacks": sum(d["eager_fallbacks"] for d in deltas),
            "all_formal_graph_shapes_warm": all(d["captures"] == 0 for d in deltas) if deltas else None,
            "steady_state_claim_allowed": bool(deltas) and sum(d["replays"] for d in deltas) > 0 and captures == 0,
            "capture_cost_included_if_present": True}
        if report["formal_graph_accounting"]["steady_state_claim_allowed"]:
            require(all(d["captures"] == 0 for d in deltas), "cannot claim steady-state with formal captures")
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


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def validate_worker(worker, *, allow_subset=False):
    require(worker["status"] == "complete" and worker["passed"], "incomplete or failed worker")
    require(worker["source"] == worker["source_end"], "source changed during benchmark")
    require(worker["source"]["git"]["status_porcelain"] == "", "dirty worker source")
    evidence = worker["actual_path"]
    require(evidence["profiler_removed_for_timed_samples"], "actual-path profiler leaked")
    if worker["mode"] == "upstream_flashattn":
        require(worker["source"]["git"]["head"] == upstream.UPSTREAM_HEAD, "not the pristine native greedy baseline")
        require(evidence["sdpa_calls"] == 0 and all(evidence["counts"].get(api, 0) > 0 for api in
                ("flash_attn_varlen_func", "flash_attn_with_kvcache")), "native FlashAttention path not proven")
    else:
        require(worker["mode"] == "jetspec", "unknown worker mode")
        require(worker["serving_policy"] == {**policy.SERVING_POLICY, "attention_backend": "sdpa"}, "serving policy changed")
        require(evidence["execution"] == worker["execution_policy"]["target_execution"] and
                evidence["kernels"] == worker["execution_policy"]["target_kernels"], "path/policy mismatch")
    names = [f"c{c['concurrency']}_o{c['output_cap_scale']}" for c in worker["cases"]]
    require(len(names) == len(set(names)) and all(name in CASE_NAMES for name in names), "invalid worker case matrix")
    require(allow_subset or tuple(names) == CASE_NAMES, "publication requires the full six-case matrix")
    repeats, warmups_per_case = worker["measurement"]["repeats"], worker["measurement"]["warmups_per_case"]
    require(repeats >= 3 and warmups_per_case >= 1, "unqualified repeat/warmup policy")
    for case in worker["cases"]:
        samples = [s for s in worker["samples"] if s["case_index"] == case["case_index"]]
        warmups = [s for s in worker["warmups"] if s["case_index"] == case["case_index"]]
        require(len(samples) == repeats and len(warmups) == warmups_per_case,
                "each case must retain every declared warmup and >=3 raw formal samples")
        require({s["repeat"] for s in samples} == set(range(repeats)), "duplicate or missing repeat ID")
        for sample in (*samples, *warmups):
            require(sample["workload_sha256"] == case["workload_sha256"], "sample workload changed")
            require(sample["exactly_once_and_cap_passed"] and sample["after_cleanup"]["used_blocks"] == 0 and
                    sample["after_cleanup"]["free_blocks"] == sample["pool"]["shape"][2], "delivery/allocator cleanup failure")
            require(sample["wall_s"] > 0 and abs(sample["tokens_per_second"] - sample["actual_output_tokens"] / sample["wall_s"]) < 1e-9,
                    "throughput is not actual tokens / wall time")
        require(all(s["profiler_active_during_timed"] is False for s in samples), "profiled formal sample")
    require(sum(s["case_index"] in {c["case_index"] for c in worker["cases"]} for s in worker["samples"]) == len(worker["samples"]),
            "unaccounted sample case")


def arm_summary(worker, case):
    samples = [s for s in worker["samples"] if s["case_index"] == case["case_index"]]
    return {"label": worker["label"], "mode": worker["mode"], "execution_policy": worker["execution_policy"],
        "raw_throughput_tok_s": [s["tokens_per_second"] for s in samples],
        "median_throughput_tok_s": median(s["tokens_per_second"] for s in samples),
        "median_wall_s": median(s["wall_s"] for s in samples),
        "median_request_metrics": {key: {stat: median(s["request_metrics"][key][stat] for s in samples)
            for stat in ("p50", "p95", "max")} for key in previous.REQUEST_METRICS},
        "median_delivery_gap_p95_s": median(s["per_request_delivery_gap_distribution_s"]["p95"] for s in samples),
        "worst_delivery_gap_max_s": max(s["per_request_delivery_gap_distribution_s"]["max"] for s in samples),
        "peak_gpu_allocated_bytes": max(s["peak_gpu_allocated_bytes"] for s in samples),
        "peak_gpu_reserved_bytes": max(s["peak_gpu_reserved_bytes"] for s in samples),
        "peak_leased_pages": max(s["peak_used_pages"] for s in samples),
        "median_emitted_tokens_per_packed_verify": median(s["mean_effective_output_block_tokens_per_packed_verify_call"] for s in samples),
        "median_emitted_tokens_per_verified_request": median(s["mean_effective_output_block_tokens_per_verified_request"] for s in samples),
        "raw_graph_deltas": [s.get("target_graph_delta") for s in samples], "allocator_cleanup_all_passed": True}


def combined_summary(baseline, candidate, old=None, *, allow_subset=False):
    workers = [baseline, *([old] if old else []), candidate]
    for w in workers:
        validate_worker(w, allow_subset=allow_subset)
    require(baseline["mode"] == "upstream_flashattn" and candidate["mode"] == "jetspec", "comparison arms swapped")
    if old:
        require(old["mode"] == "jetspec", "old arm must be JetSpec")
        require(old["models"] == candidate["models"] and old["serving_policy"] == candidate["serving_policy"], "JetSpec models/policy differ")
    for worker in workers[1:]:
        for key in ("manifest_sha256", "manifest_file_sha256", "cases", "config"):
            require(baseline[key] == worker[key], f"worker mismatch: {key}")
        require(baseline["models"]["target"] == worker["models"]["target"], "Target/tokenizer differs")
        for key in ("torch", "cuda", "gpu", "compute_capability", "executable", "python"):
            require(baseline["environment"][key] == worker["environment"][key], f"environment differs: {key}")
        for package in ("torch", "transformers", "triton", "numpy", "flash-attn"):
            require(baseline["environment"]["packages"][package] == worker["environment"]["packages"][package],
                    f"environment package differs: {package}")
    rows = []
    for case in baseline["cases"]:
        totals = {s["actual_output_tokens"] for w in workers for s in w["samples"] if s["case_index"] == case["case_index"]}
        require(len(totals) == 1, "arms emitted different actual token counts")
        arms = [arm_summary(w, case) for w in workers]
        row = {**case, "actual_output_tokens_per_sample": totals.pop(), "arms": arms,
            "jetspec_over_upstream_median_throughput_ratio": arms[-1]["median_throughput_tok_s"] / arms[0]["median_throughput_tok_s"]}
        if old:
            row["optimized_over_old_jetspec_median_throughput_ratio"] = arms[-1]["median_throughput_tok_s"] / arms[1]["median_throughput_tok_s"]
        rows.append(row)
    return rows


def combine(args):
    paths = [args.upstream_results, *([args.old_jetspec_results] if args.old_jetspec_results else []), args.jetspec_results]
    workers = [json.loads(Path(path).read_text()) for path in paths]
    rows = combined_summary(workers[0], workers[-1], workers[1] if len(workers) == 3 else None,
                            allow_subset=args.allow_subset_for_diagnostics)
    certificate = json.loads(Path(args.flash_validation).read_text())
    require(certificate["status"] == "passed" and certificate["varlen_callable"] and certificate["kvcache_callable"] and
            all(c["gpu_call_passed"] for c in certificate["cases"]), "independent native Flash GPU validation failed")
    require({c["api"] for c in certificate["cases"]} == {"flash_attn_varlen_func", "flash_attn_with_kvcache"}, "Flash certificate missed an API")
    require(tuple(certificate[key] for key in ("torch", "cuda", "gpu")) ==
            tuple(workers[0]["environment"][key] for key in ("torch", "cuda", "gpu")), "Flash certificate environment differs")
    require(certificate["flash_attn"] == workers[0]["environment"]["packages"]["flash-attn"] and
            certificate["extension_sha256"] == workers[0]["flash_extension"]["sha256"],
            "Flash certificate and worker did not use the identical native extension")
    representative = next((row for row in rows if (row["concurrency"], row["output_cap_scale"]) == (8, 512)), None)
    save(args.output, {"schema_version": 1, "status": "complete", "passed": True,
        "benchmark_type": "pristine upstream FlashAttention vs official-inspired JetSpec practical system comparison",
        "diagnostic_subset_not_resume_publication": len(rows) != 6,
        "baseline_revision_selection": {"preferred_commit": "bb823b3e06983d71485a8e1f23715ebd87d98ef8",
            "selected_commit": upstream.UPSTREAM_HEAD, "reason": "closest preceding pristine ancestor with native greedy; bb823b3 removed greedy",
            "no_upstream_production_changes": True, "not_claiming_bb823b3_was_measured": True},
        "representative_case_preregistered": {"concurrency": 8, "output_cap_scale": 512},
        "representative_median_speedup": representative["jetspec_over_upstream_median_throughput_ratio"] if representative else None,
        "summary": rows, "workers": workers,
        "raw_artifacts": [{"path": str(Path(path).resolve()), "sha256": upstream.file_sha(path)} for path in paths],
        "flash_attention_gpu_validation": {"path": str(Path(args.flash_validation).resolve()),
            "sha256": upstream.file_sha(args.flash_validation), "evidence": certificate},
        "postprocessing_harness_sha256": upstream.file_sha(__file__),
        "interpretation": {"scope": "Combined algorithm/serving/backend effects, not algorithm-only or backend-matched ablation",
            "graphs": "Upstream enforce_eager=True; candidate Target CUDA Graph is explicitly reported when selected",
            "pool": "Each system retains its natural KV allocation/memory; no geometry equality asserted",
            "output_scales": "128/512 are mixed output-cap scales, not uniform request lengths",
            "latencies": "Median of per-sample request statistics; submitted latency still includes queueing, not pure GPU time",
            "speedup": "Ratio of raw-sample throughput medians; no temporal repeat pairing and no best-run selection",
            "numerics": "Independent unchanged numerical/isolation qualification required; cross-layout bitwise token equivalence not required",
            "historical_7_27x": "Historical fork SDPA AR comparison only; never call this original nano-vLLM speedup"}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("upstream", "jetspec", "combine"), required=True)
    for flag in ("repo", "target", "draft", "manifest", "expected-head", "expected-production-sha",
                 "upstream-results", "jetspec-results", "old-jetspec-results", "flash-validation"):
        parser.add_argument("--" + flag)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--target-execution", choices=TARGET_EXECUTIONS, default="eager")
    parser.add_argument("--target-kernels", choices=TARGET_KERNELS, default="reference")
    parser.add_argument("--expected-tree-function", default="packed_tree_attention")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--deadline", type=float, default=1800)
    parser.add_argument("--cases", default="", help="diagnostic subset only; case specs remain immutable")
    parser.add_argument("--allow-subset-for-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.mode == "combine":
        require(args.upstream_results and args.jetspec_results and args.flash_validation, "combine requires workers and Flash GPU certificate")
        combine(args)
    else:
        require(all(getattr(args, field) for field in ("repo", "target", "manifest", "expected_head", "expected_production_sha")),
                "worker needs clean repo, model, manifest and source pins")
        require(args.mode == "upstream" or args.draft, "JetSpec worker requires Draft checkpoint")
        require(args.mode != "upstream" or (args.target_execution == "eager" and args.target_kernels == "reference"),
                "upstream implementation cannot be configured with JetSpec optimizations")
        worker(args)


if __name__ == "__main__":
    main()
