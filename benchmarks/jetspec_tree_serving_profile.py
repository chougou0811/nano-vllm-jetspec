#!/usr/bin/env python3
"""Real final-workload Tree Attention profile, separate from timed benchmark.

Select a clean production snapshot with --repo; this file never edits it.
The immutable final-matched workload, serving setup, offered clock, exactly-once
ledger, and allocator cleanup are reused unchanged. Profiler/event-instrumented
throughput is diagnostic only and MUST NOT be used as benchmark throughput.
"""
from __future__ import annotations

import argparse
import atexit
from collections import Counter
from contextlib import AbstractContextManager
import json
from pathlib import Path
import sys

import jetspec_flash_serving_benchmark as serving
import jetspec_phase4 as phase4
from jetspec_flash_draft_profile import summarize_trace
from jetspec_tree_profile_report import attribute_trace

upstream = serving.upstream
previous, require, save = serving.previous, serving.require, serving.save


def select_case(manifest, concurrency, output):
    cases = [c for c in manifest["cases"] if
             (c["concurrency"], c["output_cap_scale"]) == (concurrency, output)]
    require(len(cases) == 1, "requested immutable workload case absent/duplicated")
    return cases[0]


def categorize_cpu(operators):
    """Overlapping descriptive buckets, never an exclusive time partition."""
    patterns = {
        "gemm": ("aten::linear", "aten::mm", "aten::addmm", "aten::bmm"),
        "norm_and_rope_elementwise_proxy": ("aten::pow", "aten::mean", "aten::rsqrt",
                                            "aten::mul", "aten::sub", "aten::add"),
        "slot_mapping_and_scatter": ("aten::div", "aten::remainder", "aten::index_put",
                                     "aten::_index_put_impl_"),
        "cpu_materialization": ("aten::item", "aten::_local_scalar_dense"),
        "tensor_copy_and_pack": ("aten::copy_", "aten::cat", "aten::clone", "aten::contiguous",
                                 "aten::_to_copy"),
    }
    return {name: [r for r in operators if any(p == r["operator"] for p in matches)]
            for name, matches in patterns.items()}


class LateWindow(AbstractContextManager):
    """Observe engine-step return deltas; capture N actual verification rounds."""
    def __init__(self, engine, threshold, rounds, trace_path):
        import torch
        self.torch, self.engine = torch, engine
        self.threshold, self.round_limit, self.trace_path = threshold, rounds, trace_path
        self.lengths, self.rows = Counter(), []
        self.profile = None
        self.active = self.done = False
        self.old_step = None

    def __enter__(self):
        self.old_step = self.engine.step
        self.engine.step = self.step
        return self

    def step(self, *args, **kwargs):
        if not self.done and not self.active and max(self.lengths.values(), default=0) >= self.threshold:
            self.profile = self.torch.profiler.profile(
                activities=[self.torch.profiler.ProfilerActivity.CPU,
                            self.torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
            self.profile.start()
            self.active = True
        scheduler = getattr(self.engine, "_jetspec_scheduler", None)
        # Host lengths only: no tensor materialization or extra device sync.
        prefixes = {rid: int(request.state.cache_len)
                    for rid, request in scheduler.runtime.requests.items()} if scheduler else {}
        result = self.old_step(*args, **kwargs)
        info = self.engine.last_step_info
        for event in info.get("events", []):
            if event["kind"] == "tokens":
                self.lengths[event["request_id"]] += len(event["token_ids"])
        verification = info.get("verification")
        if self.active and verification:
            self.rows.append({"node_counts": list(verification.get("node_counts", [])),
                "packed_query_tokens": verification.get("total_query_tokens"),
                "prefix_lengths_before_step": [prefixes.get(rid) for rid in
                                               verification.get("request_ids", [])],
                "requests": [{"request_id": r["request_id"],
                    "emitted_tokens": len(r["output_block"]),
                    "accepted_draft_length": r.get("accepted_draft_length")}
                    for r in verification.get("requests", [])],
                "delivered_lengths_at_return": dict(self.lengths)})
            if len(self.rows) >= self.round_limit:
                self.profile.stop()
                self.active, self.done = False, True
        return result

    def __exit__(self, *exc):
        self.engine.step = self.old_step
        if self.active:
            self.profile.stop()
            self.active, self.done = False, True

    def summary(self):
        require(self.profile is not None and len(self.rows) == self.round_limit,
                "requested late verification window was not fully captured")
        self.profile.export_chrome_trace(str(self.trace_path))
        trace = json.loads(Path(self.trace_path).read_text())
        operators = sorted([{"operator": e.key, "calls": e.count,
            "self_cpu_ms": e.self_cpu_time_total / 1000,
            "total_cpu_ms": e.cpu_time_total / 1000,
            "input_shapes": e.input_shapes}
            for e in self.profile.key_averages(group_by_input_shape=True)],
            key=lambda r: -r["self_cpu_ms"])
        result = summarize_trace(trace)
        attribution = attribute_trace(trace)
        # Unlike the older Draft-only helper, include Triton's CUDA driver API
        # launches as well as runtime launches. Captured raw artifacts retain
        # their original script hash; this affects future captures only.
        result["cuda_launch_calls"] = attribution["launch_calls"]
        result["cuda_launch_call_count"] = attribution["launch_count"]
        result["cuda_synchronization_calls"] = attribution["synchronization_calls"]
        result["cuda_synchronization_call_count"] = attribution["synchronization_count"]
        result["exclusive_attribution"] = attribution
        result.update(trace_path=str(Path(self.trace_path).resolve()),
            trace_sha256=upstream.file_sha(self.trace_path),
            captured_verify_rounds=self.rows, threshold_delivered_tokens=self.threshold,
            top_cpu_operators=operators[:80], cpu_operation_categories=categorize_cpu(operators),
            phase_ranges=[r for r in operators if r["operator"].startswith("phase5.")],
            attention_cuda_activity=[r for r in result["kernel_names"]
                if any(name in r["name"] for name in ("paged_tree", "tree_attention", "packed_tree"))],
            cpu_category_note="Overlapping operator buckets; elementwise is a norm/RoPE proxy, not exclusive norm/RoPE attribution. CUDA kernels and parent phase ranges are available in Chrome trace.",
            synchronization_scope="Only bounded late serving window; profiler may introduce additional synchronization. Preparation/warmup excluded.",
            duration_note="Summed observed CUDA kernel duration is not end-to-end wall time, and overlapping ranges must not be added.")
        return result


class TargetRanges(AbstractContextManager):
    """Diagnostic-only named ranges; restore all original methods afterward."""
    def __init__(self, engine):
        import torch
        self.torch, self.engine = torch, engine
        self.patches = []

    def patch(self, obj, name, label):
        original = getattr(obj, name)
        self.patches.append((obj, name, original))
        def wrapped(*args, **kwargs):
            with self.torch.profiler.record_function(label):
                return original(*args, **kwargs)
        setattr(obj, name, wrapped)

    def __enter__(self):
        from nanovllm.models import qwen3
        from nanovllm.speculative.jetspec import paged_backend
        runtime = self.engine._jetspec_scheduler.runtime
        for name, label in (("_verify_batch", "phase5.target.verify"),
                            ("_propose_drafts", "phase5.draft"),
                            ("_build_trees", "phase5.tree_build"),
                            ("_accept_batch", "phase5.accept")):
            if hasattr(runtime, name):
                self.patch(runtime, name, label)
        self.patch(paged_backend, "packed_tree_attention", "phase5.target.tree_attention")
        self.patch(qwen3, "_reference_rms_norm", "phase5.target.reference_qk_norm")
        for module in self.engine.model_runner.model.modules():
            if isinstance(module, qwen3.Qwen3Attention):
                self.patch(module, "forward_paged_tree", "phase5.target.attention_block")
                self.patch(module.o_proj, "forward", "phase5.target.o_proj")
            elif isinstance(module, qwen3.Qwen3MLP):
                self.patch(module, "forward", "phase5.target.mlp")
                self.patch(module, "forward_dense", "phase5.target.mlp")
        self.patch(self.engine.model_runner.model.lm_head, "forward", "phase5.target.lm_head")
        return self

    def __exit__(self, *exc):
        for obj, name, original in reversed(self.patches):
            setattr(obj, name, original)


def run(args):
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch, nanovllm, jetspec
    require(torch.cuda.is_available(), "CUDA required")
    require(Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()),
            "production import escaped --repo")
    source = upstream.source_identity(nanovllm, jetspec)
    require(source["git"]["head"].startswith(args.expected_head), "wrong snapshot HEAD")
    require(source["production_sha256"] == args.expected_production_sha,
            "wrong preregistered production fingerprint")
    require(source["git"]["status_porcelain"] == "", "profile requires a clean snapshot")
    manifest = upstream.load_manifest(args.manifest)
    case = select_case(manifest, args.concurrency, args.output_scale)
    report = {"schema_version": 1, "status": "in_progress", "passed": False,
        "source": source, "script_sha256": upstream.file_sha(__file__),
        "helper_sha256": {Path(p).name: upstream.file_sha(p) for p in
            (serving.__file__, upstream.__file__, previous.__file__, phase4.__file__)},
        "environment": upstream.environment(torch), "config": serving.USER_CONFIG,
        "models": {"target": previous._helpers.checkpoint(args.target),
                   "draft": previous._helpers.checkpoint(args.draft)},
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_file_sha256": upstream.file_sha(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "case": {k: v for k, v in case.items() if k != "specs"},
        "policy": {**serving.SERVING_POLICY, "attention_backend": "sdpa"},
        "diagnostic_only": True,
        "timing_contract": "All profiled/event-instrumented throughput is excluded from performance claims. Matched serving throughput must be collected separately with existing benchmark worker.",
        "invocation": {"argv": sys.argv, "arguments": vars(args)}}
    save(args.output, report)
    engine = None
    try:
        torch.manual_seed(0)
        engine = nanovllm.LLM(args.target, **serving.USER_CONFIG)
        runtime = engine.get_jetspec_batch_runtime(args.draft)
        report["pool"] = previous.pool_identity(engine.model_runner.kv_cache)
        report["pool"]["block_size"] = engine.scheduler.block_manager.block_size
        serving.prepare_candidate(engine, runtime, args.draft, args.concurrency, "sdpa")
        print("immutable workload warmup START", flush=True)
        report["warmup"] = previous.serve(engine, runtime, case, "jetspec", args.deadline)
        save(args.output, report)
        serving.prepare_candidate(engine, runtime, args.draft, args.concurrency, "sdpa")
        print("bounded late kernel profile START", flush=True)
        trace_path = Path(args.output).with_suffix(".trace.json")
        with TargetRanges(engine), LateWindow(engine, args.threshold, args.rounds, trace_path) as window:
            report["kernel_window_run"] = previous.serve(engine, runtime, case, "jetspec", args.deadline)
        report["kernel_window"] = window.summary()
        save(args.output, report)
        if not args.skip_stage:
            serving.prepare_candidate(engine, runtime, args.draft, args.concurrency, "sdpa")
            print("full workload CUDA-event stage profile START", flush=True)
            with phase4.StageProbe(engine) as probe:
                report["stage_run"] = previous.serve(engine, runtime, case, "jetspec", args.deadline)
                report["stage_profile"] = probe.metrics()
            save(args.output, report)
        require(upstream.source_identity(nanovllm, jetspec) == source,
                "production source changed during profiling")
        report.update(status="complete", passed=True, source_end=source)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "target", "draft", "manifest", "output", "expected-head", "expected-production-sha"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 4, 8), default=8)
    parser.add_argument("--output-scale", type=int, choices=(128, 512), default=512)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--deadline", type=float, default=300)
    parser.add_argument("--skip-stage", action="store_true")
    args = parser.parse_args()
    require(args.threshold >= 0 and args.rounds > 0 and args.deadline > 0,
            "invalid profile window/deadline")
    run(args)


if __name__ == "__main__":
    main()
