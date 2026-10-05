#!/usr/bin/env python3
"""Untimed attribution + small same-input Draft microtimings, NOT serving data.

Run after the serving workers have exited. Both proposers use the same trained
weights and cloned states obtained from one real SDPA accepted-only Target
commit. No production function/kernel or grouping parameter is replaced.
"""
from __future__ import annotations

import argparse
import atexit
from collections import Counter
from contextlib import AbstractContextManager, contextmanager
import json
from pathlib import Path
import statistics
import sys
import time

import jetspec_final_upstream_flashattn as provenance
import jetspec_phase4_qualification as frozen

require, save = provenance.require, provenance.save
EXPECTED_HEAD = "6f65f6e"
MODES = ("sdpa", "flash_attn")
PACKING_OPS = {"aten::cat", "aten::copy_", "aten::clone", "aten::contiguous"}
LINEAR_OPS = {"aten::linear", "aten::mm", "aten::addmm", "aten::bmm"}


def summarize_microtimings(rows):
    result = {}
    for mode in MODES:
        runs = [row for row in rows if row["mode"] == mode and row["phase"] == "timed"]
        require(len(runs) == 10, "need exactly ten retained microtimings per mode")
        require(len({row["repeat"] for row in runs}) == 10, "duplicate microtiming repeat")
        result[mode] = {"samples": len(runs),
            "median_cpu_wall_ms_including_tail_sync": statistics.median(row["cpu_wall_ms"] for row in runs),
            "median_cuda_event_timeline_ms": statistics.median(row["cuda_event_ms"] for row in runs),
            "raw_cpu_wall_ms": [row["cpu_wall_ms"] for row in runs],
            "raw_cuda_event_ms": [row["cuda_event_ms"] for row in runs]}
    result["interpretation"] = (
        "Draft-only repeated same-input microtimings; cloning/preparation excluded. "
        "CUDA event elapsed time includes GPU stream idle gaps caused by host enqueue delays, "
        "not just summed kernel execution. CPU wall includes completion synchronization. "
        "Not aggregate serving throughput, and no serving speedup may be inferred from these ratios.")
    return result


def summarize_trace(trace):
    """Chrome Kineto events: actual GPU kernels and runtime API counts/times."""
    events = [row for row in trace.get("traceEvents", []) if row.get("ph") == "X"]
    kernels, runtimes = Counter(), Counter()
    kernel_us, runtime_us = Counter(), Counter()
    for row in events:
        name, category = row.get("name", ""), row.get("cat", "")
        if category == "kernel":
            kernels[name] += 1
            kernel_us[name] += float(row.get("dur", 0))
        elif category == "cuda_runtime":
            runtimes[name] += 1
            runtime_us[name] += float(row.get("dur", 0))
    launches = {name: count for name, count in runtimes.items() if "LaunchKernel" in name}
    syncs = {name: count for name, count in runtimes.items() if "Synchronize" in name}
    return {"cuda_kernel_count": sum(kernels.values()),
        "summed_cuda_kernel_duration_ms": sum(kernel_us.values()) / 1000,
        "kernel_names": [{"name": name, "calls": kernels[name], "total_gpu_ms": kernel_us[name] / 1000}
            for name in sorted(kernels, key=lambda name: -kernel_us[name])],
        "cuda_launch_calls": launches, "cuda_launch_call_count": sum(launches.values()),
        "cuda_synchronization_calls": syncs, "cuda_synchronization_call_count": sum(syncs.values()),
        "cuda_runtime": [{"name": name, "calls": runtimes[name], "total_cpu_ms": runtime_us[name] / 1000}
            for name in sorted(runtimes, key=lambda name: -runtime_us[name])],
        "synchronization_scope": "includes the one explicitly labelled profile tail cudaDeviceSynchronize; no cloning or preparation"}


def attention_operators(rows):
    return [row for row in rows if "scaled_dot_product" in row["name"] or
            "flash_attn" in row["name"] or "FlashAttn" in row["name"]]


class ProposalCallEvidence(AbstractContextManager):
    """First warmup only, original FA wrapper code and built-in SDPA calls."""
    def __init__(self, function):
        self.function = function
        self.count = self.sdpa_calls = 0

    def _profile(self, frame, event, arg):
        if event == "call" and frame.f_code is self.function.__code__:
            self.count += 1
        elif event == "c_call" and getattr(arg, "__name__", "") == "scaled_dot_product_attention":
            self.sdpa_calls += 1

    def __enter__(self):
        require(sys.getprofile() is None, "another profiler is active")
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)

    def result(self, mode):
        require(self.count > 0 if mode == "flash_attn" else self.count == 0,
                "original external FlashAttention calls disagree with requested mode")
        return {"original_flash_attn_varlen_calls": self.count, "sdpa_calls": self.sdpa_calls,
            "function_module": self.function.__module__, "function_file": self.function.__code__.co_filename,
            "method": "untimed first warmup sys.setprofile; no original function replacement",
            "profiler_removed": sys.getprofile() is None}


@contextmanager
def module_regions(runtime):
    """Profile-only annotations; hooks add overhead, so never enter microtimings."""
    import torch
    hooks, active = [], {}
    modules = [(runtime.head.fc, "Draft::target_feature_fc"),
               (runtime.target.lm_head, "Draft::lm_head")]
    for layer in runtime.head.layers:
        modules.append((layer.mlp, "Draft::MLP"))
        modules += [(getattr(layer.self_attn, name), "Draft::QKV_O_projections")
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
    def begin(module, args):
        scope = torch.profiler.record_function(active[module]["name"])
        active[module]["stack"].append(scope)
        scope.__enter__()
    def end(module, args, output):
        active[module]["stack"].pop().__exit__(None, None, None)
    try:
        for module, name in modules:
            active[module] = {"name": name, "stack": []}
            hooks.extend((module.register_forward_pre_hook(begin),
                          module.register_forward_hook(end, always_call=True)))
        yield
    finally:
        for hook in hooks:
            hook.remove()
        for entry in active.values():
            while entry["stack"]:
                entry["stack"].pop().__exit__(None, None, None)


def operator_rows(profiler):
    rows = []
    for event in profiler.key_averages():
        rows.append({"name": event.key, "calls": event.count,
            "self_cpu_ms": event.self_cpu_time_total / 1000,
            "inclusive_cpu_ms": event.cpu_time_total / 1000,
            "self_gpu_ms": getattr(event, "self_device_time_total", 0) / 1000,
            "inclusive_gpu_ms": getattr(event, "device_time_total", 0) / 1000})
    return sorted(rows, key=lambda row: -row["self_cpu_ms"])


def profile_once(proposer, runtime, requests, depth, mode, output):
    import torch
    cloned = [frozen.clone_request(runtime, request) for request in requests]
    torch.cuda.synchronize()  # Clone/cache preparation is completely outside capture.
    with module_regions(runtime), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=False, with_stack=False) as profiler:
        with torch.profiler.record_function("Draft::proposal_only"):
            result = proposer.propose(cloned, depth)
        with torch.profiler.record_function("Profile::explicit_tail_completion_sync"):
            torch.cuda.synchronize()
    trace_path = str(Path(output).with_suffix(f".{mode}.trace.json"))
    profiler.export_chrome_trace(trace_path)
    rows = operator_rows(profiler)
    trace = summarize_trace(json.loads(Path(trace_path).read_text()))
    require(trace["cuda_kernel_count"] > 0, "CUDA profiler did not capture actual kernels")
    answer = {"mode": mode, "trace_path": trace_path, "trace_sha256": provenance.file_sha(trace_path),
        "attention_dispatch_operators": attention_operators(rows), "operators": rows,
        "module_regions": [row for row in rows if row["name"].startswith("Draft::")],
        "packing_copy_operator_proxy": [row for row in rows if row["name"] in PACKING_OPS],
        "linear_operator_proxy": [row for row in rows if row["name"] in LINEAR_OPS],
        "packing_proxy_caveat": "cat/copy/clone/contiguous include KV packing but also noise/RoPE/metadata operations; not exclusive KV attribution",
        "annotation_caveat": "module hooks exist only during this untimed profile; their overhead and profiler timings are not serving or microtiming results",
        "proposer_stats": dict(proposer.last_stats), **trace}
    del result, cloned
    return answer


def run(args):
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch, nanovllm, jetspec, flash_attn
    from nanovllm.speculative.jetspec.batched_draft import BatchedDraftProposer
    from nanovllm.speculative.jetspec.flash_draft import FlashDraftProposer
    require(torch.cuda.is_available(), "CUDA is required")
    source = provenance.source_identity(nanovllm, jetspec)
    require(Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()), "wrong production import")
    require(source["git"]["head"].startswith(EXPECTED_HEAD) and source["git"]["status_porcelain"] == "",
            "profile requires the clean pinned 6f65f6e production snapshot")
    report = {"kind": "same-input trained Draft microtiming and untimed kernel attribution",
        "status": "in_progress", "passed": False, "NOT_A_SERVING_BENCHMARK": True,
        "source": source, "profile_script_sha256": provenance.file_sha(__file__),
        "environment": provenance.environment(torch),
        "models": {"target": provenance.previous._helpers.checkpoint(args.target),
                   "draft": provenance.previous._helpers.checkpoint(args.draft)},
        "policy": {"concurrency": 8, "prompt_lengths": [1024, 2048] * 4,
            "tree_budgets": [63, 31, 47, 63, 31, 47, 63, 31], "max_output_tokens": 128,
            "warmups_per_mode": 3, "timed_repeats_per_mode": 10, "max_padding_ratio": 2.0,
            "same_base_state_for_every_proposal": True, "clone_outside_timing": True,
            "microtiming_order": "alternating mode order by repeat after all warmups",
            "prefix_and_real_commit_backend": "sdpa Draft + unchanged packed Triton Target",
            "chunked_prefill": False}, "warmups": [], "microtimings": [], "profiles": []}
    engine = runtime = None
    requests = []
    save(args.output, report)
    try:
        torch.manual_seed(0)
        engine = nanovllm.LLM(args.target, tensor_parallel_size=1, enforce_eager=True,
            max_num_seqs=8, max_model_len=4096, max_num_batched_tokens=4096,
            gpu_memory_utilization=.8, kvcache_block_size=256)
        runtime = engine.get_jetspec_batch_runtime(args.draft)
        runtime.configure_optimizations(lightweight=True, batched_draft=True,
            feature_storage=True, attention_backend="sdpa")
        require(runtime._prefill_attention_backend == "sdpa", "unqualified prefill backend entered profile")
        for i, length in enumerate(report["policy"]["prompt_lengths"]):
            ids = provenance.previous._helpers.prompt(engine.tokenizer, length, i)
            requests.append(runtime.create_request(ids, max_new_tokens=128,
                tree_budget=(63, 31, 47)[i % 3], ignore_eos=True, request_id=f"profile:{i}"))
        commit = runtime.step(requests)
        for request in requests:
            request.state.assert_round_invariant(validate_device=True)
        report["base_state"] = [{"request_id": r.request_id,
            "prompt_length": r.prompt_length, "tree_budget": r.tree_budget,
            "context_length": int(r.state.target_hidden.shape[1]),
            "cached_draft_length": r.drafter._fwd.cache.get_seq_length(),
            "new_context_suffix": int(r.state.target_hidden.shape[1]) - r.drafter._fwd.cache.get_seq_length(),
            "emitted_output_tokens": len(r.output_ids)} for r in requests]
        require(all(row["cached_draft_length"] > 0 and row["new_context_suffix"] > 0
                    for row in report["base_state"]), "real commit did not produce warm cache plus new suffix")
        report["base_commit_packed_requests"] = len(commit["requests"])
        proposers = {"sdpa": BatchedDraftProposer(runtime.head, runtime.target),
                     "flash_attn": FlashDraftProposer(runtime.head, runtime.target)}
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            for mode in MODES:
                for repeat in range(3):
                    cloned = [frozen.clone_request(runtime, request) for request in requests]
                    torch.cuda.synchronize()
                    if repeat == 0:
                        with ProposalCallEvidence(flash_attn.flash_attn_varlen_func) as evidence:
                            outputs = proposers[mode].propose(cloned, runtime.tree_depth)
                        report.setdefault("actual_path_evidence", {})[mode] = evidence.result(mode)
                    else:
                        outputs = proposers[mode].propose(cloned, runtime.tree_depth)
                    torch.cuda.synchronize()
                    report["warmups"].append({"mode": mode, "repeat": repeat,
                                              "proposer_stats": dict(proposers[mode].last_stats)})
                    del outputs, cloned
            for repeat in range(10):
                for mode in MODES if repeat % 2 == 0 else reversed(MODES):
                    cloned = [frozen.clone_request(runtime, request) for request in requests]
                    torch.cuda.synchronize()
                    start_event.record()
                    started = time.perf_counter()
                    outputs = proposers[mode].propose(cloned, runtime.tree_depth)
                    end_event.record()
                    end_event.synchronize()
                    report["microtimings"].append({"phase": "timed", "mode": mode, "repeat": repeat,
                        "cpu_wall_ms": (time.perf_counter() - started) * 1000,
                        "cuda_event_ms": start_event.elapsed_time(end_event),
                        "proposer_stats": dict(proposers[mode].last_stats)})
                    del outputs, cloned
            report["microtiming_summary"] = summarize_microtimings(report["microtimings"])
            save(args.output, report)
            for mode in MODES:
                report["profiles"].append(profile_once(proposers[mode], runtime, requests,
                    runtime.tree_depth, mode, args.output))
                save(args.output, report)
        require(provenance.source_identity(nanovllm, jetspec) == source, "production changed during profile")
        for request in requests:
            runtime.cancel(request)
        runtime.release_idle_scratch()
        report["allocator_clean"] = not engine.scheduler.block_manager.used_block_ids
        require(report["allocator_clean"], "profile retained KV pages")
        report.update(status="complete", passed=True)
        save(args.output, report)
        print(json.dumps(report["microtiming_summary"], indent=2), flush=True)
    except BaseException as error:
        report.update(status="failed", passed=False, failure=f"{type(error).__name__}: {error}")
        save(args.output, report)
        raise
    finally:
        if runtime is not None:
            for request in requests:
                if runtime.requests.get(request.request_id) is request:
                    runtime.cancel(request)
            runtime.release_idle_scratch()
        if engine is not None:
            atexit.unregister(engine.exit)
            engine.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
