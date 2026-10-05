#!/usr/bin/env python3
"""Independent, same-input Phase-5 paged ragged tree operator benchmark.

No model, serving policy, or tree budget is modified. The legacy TILE=64
FP32 implementation remains an explicit baseline even after the serving
dispatcher changes. Preparation, CPU FP64 references, and profiling are all
outside timing. CUDA events measure a repeated operator GPU timeline span;
they may include host launch gaps and are not a claimed sum of kernel time.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

FP32_BOUND = 1e-4
COUNTS = (63, 31, 47)


@dataclass(frozen=True)
class Shape:
    name: str
    prefixes: tuple[int, ...]
    nodes: tuple[int, ...]
    block_size: int = 256
    kv_heads: int = 8
    groups: int = 4
    head_size: int = 128

    def validate(self):
        if not self.prefixes or len(self.prefixes) != len(self.nodes):
            raise ValueError("request prefixes and tree sizes must align")
        if any(p < 0 for p in self.prefixes) or any(n < 1 for n in self.nodes):
            raise ValueError("invalid prefix or tree length")
        if min(self.block_size, self.kv_heads, self.groups, self.head_size) < 1:
            raise ValueError("invalid page/head geometry")


def cases():
    result = [Shape(f"c{c}_p{p}", (p,) * c, tuple(COUNTS[i % 3] for i in range(c)))
              for c in (1, 4, 8) for p in (128, 1024, 2048)]
    result.append(Shape("c8_mixed_prefix", (128, 1024, 2048, 0, 257, 255, 256, 2049),
                        tuple(COUNTS[i % 3] for i in range(8))))
    return result


def binary_parents(nodes):
    if nodes < 1:
        raise ValueError("a tree requires a root")
    return (-1,) + tuple((i - 1) // 2 for i in range(1, nodes))


def chain(parents, row):
    if not parents or parents[0] != -1 or not 0 <= row < len(parents):
        raise ValueError("invalid rooted tree or chosen row")
    answer = []
    while row != -1:
        if row in answer or not 0 <= row < len(parents):
            raise ValueError("cyclic or invalid parent chain")
        answer.append(row)
        parent = parents[row]
        if parent >= row:
            raise ValueError("parents must precede children")
        row = parent
    if answer[-1] != 0:
        raise ValueError("disconnected parent chain")
    return tuple(reversed(answer))


def layout(shape, seed):
    """Host ownership proof with random non-contiguous physical pages/slots."""
    shape.validate()
    rng = random.Random(seed)
    page_counts = [(p + shape.block_size - 1) // shape.block_size for p in shape.prefixes]
    scratch_count = (sum(shape.nodes) + shape.block_size - 1) // shape.block_size + 2
    page_ids = list(range(sum(page_counts) + scratch_count + 3))
    rng.shuffle(page_ids)
    cursor, tables = 0, []
    for count in page_counts:
        tables.append(tuple(page_ids[cursor:cursor + count]))
        cursor += count
    scratch_pages = tuple(page_ids[cursor:cursor + scratch_count])
    scratch_slots = [p * shape.block_size + offset for p in scratch_pages
                     for offset in range(shape.block_size)]
    rng.shuffle(scratch_slots)
    tree_slots, cursor = [], 0
    for count in shape.nodes:
        tree_slots.append(tuple(scratch_slots[cursor:cursor + count]))
        cursor += count
    owned_prefix = {p for table in tables for p in table}
    if owned_prefix.intersection(scratch_pages) or len(set(sum(tree_slots, ()))) != sum(shape.nodes):
        raise AssertionError("layout leaked page/slot ownership")
    return {"prefix_pages": tuple(tables), "tree_slots": tuple(tree_slots),
            "scratch_pages": scratch_pages, "num_pages": len(page_ids),
            "parents": tuple(binary_parents(n) for n in shape.nodes)}


def metadata(shape, ownership, device):
    import torch
    from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata
    masks = []
    for parents in ownership["parents"]:
        mask = torch.zeros((len(parents), len(parents)), dtype=torch.bool)
        for row in range(len(parents)):
            mask[row, list(chain(parents, row))] = True
        masks.append(mask.to(device))
    slots = [torch.tensor(values, dtype=torch.int64, device=device)
             for values in ownership["tree_slots"]]
    return PackedTreeMetadata.build(shape.prefixes, ownership["prefix_pages"], slots,
                                    masks, shape.block_size), slots


def prepare(shape, seed):
    import torch
    ownership = layout(shape, seed)
    meta, slots = metadata(shape, ownership, "cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((sum(shape.nodes), shape.kv_heads * shape.groups, shape.head_size),
                    generator=generator, device="cuda", dtype=torch.bfloat16)
    pool_shape = (ownership["num_pages"], shape.block_size, shape.kv_heads, shape.head_size)
    k = torch.randn(pool_shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(pool_shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    return q, k, v, meta, ownership, slots


def legacy(q, k, v, meta, scale, groups, *, output_dtype=None):
    """Call the frozen old kernel, bypassing any newer serving dispatcher."""
    import torch
    import triton
    from nanovllm.speculative.jetspec.paged_backend import _packed_paged_tree_fp32
    out = torch.empty_like(q, dtype=output_dtype or q.dtype)
    _packed_paged_tree_fp32[(q.shape[0], q.shape[1])](
        out, q, k, v, meta.query_to_request, meta.query_local_row, meta.prefix_lens,
        meta.node_counts, meta.cu_seqlens_q, meta.block_tables, meta.tree_slots,
        meta.qq_bias, meta.qq_bias_offsets, scale,
        q_stride_0=q.stride(0), q_stride_1=q.stride(1),
        out_stride_0=out.stride(0), out_stride_1=out.stride(1),
        table_stride_0=meta.block_tables.stride(0),
        k_stride_0=k.stride(0), k_stride_1=k.stride(1), k_stride_2=k.stride(2), k_stride_3=k.stride(3),
        v_stride_0=v.stride(0), v_stride_1=v.stride(1), v_stride_2=v.stride(2), v_stride_3=v.stride(3),
        num_queries_per_kv=groups, block_size=meta.block_size, head_size=q.shape[-1],
        BLOCK_D=triton.next_power_of_2(q.shape[-1]), TILE=64)
    return out


def resolve(spec):
    module, separator, function = spec.partition(":")
    if not separator or not module or not function:
        raise ValueError("candidate must be module:function")
    result = getattr(importlib.import_module(module), function)
    if not callable(result):
        raise ValueError("candidate is not callable")
    return result


def function_identity(function):
    target = getattr(function, "fn", function)
    file = Path(inspect.getsourcefile(target)).resolve()
    return {"module": target.__module__, "function": target.__name__, "file": str(file),
            "file_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "function_sha256": hashlib.sha256(inspect.getsource(target).encode()).hexdigest()}


def source_identity(repo):
    root = Path(repo).resolve()
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted((root / "nanovllm").rglob("*.py"))}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"root": str(root), "head": git("rev-parse", "HEAD"),
            "status_porcelain": git("status", "--porcelain"), "production_files": files,
            "production_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def prefix_slots(shape, ownership, request):
    return tuple(ownership["prefix_pages"][request][position // shape.block_size] * shape.block_size
                 + position % shape.block_size for position in range(shape.prefixes[request]))


def oracle_rows(q, k, v, shape, ownership):
    """Independent selected-query CPU FP64 reference, no qq_bias inspection."""
    import torch
    from jetspec_numeric_oracles import fp64_attention
    q_cpu, k_cpu, v_cpu = (value.detach().cpu() for value in (q, k, v))
    answer, offset = [], 0
    for request, count in enumerate(shape.nodes):
        for local in sorted({0, count // 2, count - 1}):
            selected = prefix_slots(shape, ownership, request) + tuple(
                ownership["tree_slots"][request][node] for node in chain(ownership["parents"][request], local))
            addresses = torch.tensor(selected, dtype=torch.int64)
            reference = fp64_attention(q_cpu[offset + local],
                k_cpu[addresses // shape.block_size, addresses % shape.block_size],
                v_cpu[addresses // shape.block_size, addresses % shape.block_size],
                shape.head_size ** -0.5, shape.groups)
            answer.append((offset + local, reference))
        offset += count
    return answer


def error_metrics(actual, reference):
    import torch
    left, right = actual.detach().cpu().double(), reference.detach().cpu().double()
    if left.shape != right.shape or left.numel() < 1:
        raise ValueError("error metrics require matching nonempty tensors")
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        return {"passed": False, "nonfinite": True}
    delta = left - right
    max_abs = float(delta.abs().max())
    rms = float(delta.square().mean().sqrt())
    ref_rms = float(right.square().mean().sqrt())
    relative = rms / ref_rms if ref_rms else (0 if not rms else math.inf)
    max_bound = FP32_BOUND * max(1, float(right.abs().max()))
    return {"max_abs": max_abs, "relative_rms": relative, "scaled_max_bound": max_bound,
            "relative_rms_bound": FP32_BOUND, "passed": max_abs <= max_bound and relative <= FP32_BOUND,
            "bitwise_equal": bool(torch.equal(actual.detach().cpu(), reference.detach().cpu()))}


def validate(function, tensors, shape):
    import torch
    q, k, v, meta, ownership, _ = tensors
    scale = shape.head_size ** -0.5
    # Keep native BF16 Q/K/V and the same dispatch; change output STORE only.
    candidate = function(q, k, v, meta, scale, shape.groups, output_dtype=torch.float32)
    baseline = legacy(q, k, v, meta, scale, shape.groups, output_dtype=torch.float32)
    torch.cuda.synchronize()
    rows = oracle_rows(q, k, v, shape, ownership)
    checks = []
    for row, expected in rows:
        checks.append({"global_row": row, "baseline": error_metrics(baseline[row], expected),
                       "candidate": error_metrics(candidate[row], expected)})
    # Fixed-shape isolation, not cross-shape bitwise token equivalence.
    original = function(q, k, v, meta, scale, shape.groups)
    changed_q, changed_k, changed_v = q.clone(), k.clone(), v.clone()
    selected = prefix_slots(shape, ownership, 0) + ownership["tree_slots"][0]
    addresses = torch.tensor(selected, device=q.device, dtype=torch.int64)
    changed_k[addresses // shape.block_size, addresses % shape.block_size] *= -17
    changed_v[addresses // shape.block_size, addresses % shape.block_size] += 31
    changed_q[meta.request_slice(0)] *= 7
    changed = function(changed_q, changed_k, changed_v, meta, scale, shape.groups)
    neighbors = [bool(torch.equal(original[meta.request_slice(i)], changed[meta.request_slice(i)]))
                 for i in range(1, len(shape.nodes))]
    chosen = shape.nodes[0] - 1
    ancestors = set(chain(ownership["parents"][0], chosen))
    offbranch = [slot for node, slot in enumerate(ownership["tree_slots"][0]) if node not in ancestors]
    changed_k, changed_v = k.clone(), v.clone()
    addresses = torch.tensor(offbranch, device=q.device, dtype=torch.int64)
    changed_k[addresses // shape.block_size, addresses % shape.block_size] *= 13
    changed_v[addresses // shape.block_size, addresses % shape.block_size] -= 19
    changed = function(q, changed_k, changed_v, meta, scale, shape.groups)
    branch_equal = bool(torch.equal(original[chosen], changed[chosen]))
    repeat = function(q, k, v, meta, scale, shape.groups)
    repeat_equal = bool(torch.equal(original, repeat))
    passed = (all(row[arm]["passed"] for row in checks for arm in ("baseline", "candidate"))
              and all(neighbors) and branch_equal and repeat_equal)
    return {"passed": passed, "fp32_pre_bf16_bound": FP32_BOUND,
            "oracle_selected_rows_per_request": "root/middle/last; independent CPU FP64 parent chain",
            "oracle_checks": checks, "all_rows_candidate_vs_legacy_fp32": error_metrics(candidate, baseline),
            "cross_request_fixed_shape_byte_equal": neighbors,
            "finite_offbranch_perturbation_byte_equal": branch_equal,
            "repeat_byte_equal": repeat_equal,
            "serving_bf16_output_sha256": hashlib.sha256(original.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}


def sample(function, tensors, shape, iterations):
    import torch
    q, k, v, meta = tensors[:4]
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        output = function(q, k, v, meta, shape.head_size ** -0.5, shape.groups)
    end.record()
    end.synchronize()
    wall = time.perf_counter() - start_wall
    return {"cuda_event_span_ms_per_call": begin.elapsed_time(end) / iterations,
            "synchronous_wall_ms_per_call": wall * 1000 / iterations,
            "iterations": iterations, "output_dtype": str(output.dtype), "output_shape": list(output.shape)}


def summarize(samples):
    if len(samples) < 3:
        raise ValueError("at least three formal samples are required")
    return {"samples": len(samples),
            "median_cuda_event_span_ms_per_call": statistics.median(s["cuda_event_span_ms_per_call"] for s in samples),
            "median_synchronous_wall_ms_per_call": statistics.median(s["synchronous_wall_ms_per_call"] for s in samples)}


def compiled_kernel_evidence():
    """Loaded Triton compiler/driver metadata, NOT hardware profiler counters."""
    result, seen = [], set()
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("nanovllm.speculative.jetspec.") or module is None:
            continue
        for name, function in list(vars(module).items()):
            caches = getattr(function, "device_caches", None)
            if caches is None:
                continue
            for device, state in list(caches.items()):
                for kernel in list(state[0].values()):
                    digest = getattr(kernel, "hash", None)
                    if digest in seen:
                        continue
                    seen.add(digest)
                    metadata = getattr(kernel, "metadata", None)
                    result.append({"python_module": module_name, "python_symbol": name,
                        "device": device, "compiled_hash": digest, "name": getattr(kernel, "name", None),
                        "n_regs": getattr(kernel, "n_regs", None), "n_spills": getattr(kernel, "n_spills", None),
                        "n_max_threads": getattr(kernel, "n_max_threads", None),
                        "shared_bytes": getattr(metadata, "shared", None),
                        "num_warps": getattr(metadata, "num_warps", None),
                        "num_stages": getattr(metadata, "num_stages", None),
                        "scope": "all JIT variants compiled/loaded in this worker through this case, including FP32-output diagnostics",
                        "not_hardware_counter": True})
    return result


def profile(function, tensors, shape, arm, output):
    import torch
    q, k, v, meta = tensors[:4]
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=False) as capture:
        with torch.profiler.record_function(f"TreeAttention::{arm}"):
            result = function(q, k, v, meta, shape.head_size ** -0.5, shape.groups)
        torch.cuda.synchronize()
    trace = Path(output).with_suffix(f".{shape.name}.{arm}.trace.json")
    capture.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel" and "dur" in event]
    rows = [{"name": event.key, "calls": event.count,
             "self_cpu_ms": event.self_cpu_time_total / 1000,
             "self_gpu_ms": getattr(event, "self_device_time_total", 0) / 1000}
            for event in capture.key_averages()]
    return {"trace_path": str(trace), "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "kernel_count": len(kernels), "sum_kernel_ms": sum(event["dur"] for event in kernels) / 1000,
            "kernels": [{"name": event["name"], "duration_us": event["dur"]} for event in kernels],
            "cuda_launch_count": sum(event.get("name") == "cudaLaunchKernel" for event in events),
            "cuda_stream_synchronize_count": sum(event.get("name") == "cudaStreamSynchronize" for event in events),
            "operators": rows, "profiler_active_during_formal_samples": False,
            "note": "one untimed capture; kernel durations are not serving throughput or formal sample medians"}


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(args):
    chosen = [shape for shape in cases() if not args.cases or shape.name in args.cases.split(",")]
    if not chosen:
        raise ValueError("no requested cases found")
    if args.mode == "plan":
        result = {"cases": [{"shape": asdict(shape), "layout": layout(shape, args.seed + i)}
                            for i, shape in enumerate(chosen)]}
        if args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, indent=2))
        return result
    if args.repeats < 3 or args.warmups < 1 or args.iterations < 1:
        raise ValueError("micro needs at least one warmup and three formal samples")
    if not args.output:
        raise ValueError("micro requires --output to preserve all raw samples")
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    import triton
    import nanovllm
    from nanovllm.speculative.jetspec.paged_backend import _packed_paged_tree_fp32
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU timing fallback")
    if not Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()):
        raise RuntimeError("production import did not resolve to --repo")
    candidate_function = resolve(args.candidate)
    variant = {key: value for key, value in {"query_tile": args.query_tile,
        "num_warps": args.num_warps, "maxnreg": getattr(args, "maxnreg", None)}.items()
        if value is not None}
    function = partial(candidate_function, **variant)
    baseline_identity = function_identity(_packed_paged_tree_fp32)
    baseline_identity["launch_contract"] = "FP32 multiply/reduction/online softmax; TILE=64; one(query,head) program"
    report = {"kind": "Phase-5 same-input synthetic paged ragged tree operator microbenchmark",
        "NOT_A_SERVING_BENCHMARK": True, "status": "in_progress", "passed": False,
        "source": source_identity(args.repo), "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline": baseline_identity, "candidate": function_identity(candidate_function),
        "candidate_variant": variant,
        "candidate_spec": args.candidate, "seed": args.seed,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
            "gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
            "python": sys.version, "executable": sys.executable},
        "measurement": {"warmups_per_arm": args.warmups, "repeats": args.repeats,
            "iterations_per_sample": args.iterations, "sample_order": "alternating baseline/candidate by repeat",
            "event_span_caveat": "CUDA events include GPU idle if host cannot enqueue fast enough; separate untimed profiler kernel durations",
            "no_policy_or_precision_tuning": True, "input_dtype": "torch.bfloat16",
            "validation_precision": "native BF16 Q/K/V; output_dtype=float32 changes store only, no input promotion"},
        "cases": []}
    write_json(args.output, report)
    for i, shape in enumerate(chosen):
        print(f"prepare {shape.name}", flush=True)
        tensors = prepare(shape, args.seed + i)
        validation = validate(function, tensors, shape)
        result = {"shape": asdict(shape), "metadata": tensors[3].report(),
                  "layout": tensors[4], "validation": validation,
                  "warmups": {"baseline": [], "candidate": []}, "samples": {"baseline": [], "candidate": []}}
        report["cases"].append(result)
        if not validation["passed"]:
            report.update(status="failed", failure=f"fixed numerical/isolation gate failed: {shape.name}")
            write_json(args.output, report)
            raise AssertionError(report["failure"])
        for _ in range(args.warmups):
            for arm, impl in (("baseline", legacy), ("candidate", function)):
                result["warmups"][arm].append(sample(impl, tensors, shape, args.iterations))
        for repeat in range(args.repeats):
            order = (("baseline", legacy), ("candidate", function))
            if repeat % 2:
                order = order[::-1]
            for arm, impl in order:
                entry = sample(impl, tensors, shape, args.iterations)
                entry.update(repeat=repeat, arm=arm)
                result["samples"][arm].append(entry)
        result["summary"] = {arm: summarize(samples) for arm, samples in result["samples"].items()}
        result["summary"]["cuda_event_span_speedup"] = (
            result["summary"]["baseline"]["median_cuda_event_span_ms_per_call"] /
            result["summary"]["candidate"]["median_cuda_event_span_ms_per_call"])
        if args.profile and (not args.profile_cases or shape.name in args.profile_cases.split(",")):
            result["profile"] = {arm: profile(impl, tensors, shape, arm, args.output)
                                 for arm, impl in (("baseline", legacy), ("candidate", function))}
        torch.cuda.synchronize()
        result["compiled_kernel_evidence"] = compiled_kernel_evidence()
        print(f"{shape.name} GPU event speedup={result['summary']['cuda_event_span_speedup']:.3f}", flush=True)
        write_json(args.output, report)
        del tensors
    report["source_end"] = source_identity(args.repo)
    report["source_unchanged"] = report["source"]["production_sha256"] == report["source_end"]["production_sha256"]
    report.update(status="complete", passed=report["source_unchanged"])
    write_json(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "micro"), default="micro")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--candidate", default="nanovllm.speculative.jetspec.tree_attention:packed_tree_attention_gqa")
    parser.add_argument("--query-tile", choices=(1, 2, 4), type=int)
    parser.add_argument("--num-warps", choices=(4, 8), type=int)
    parser.add_argument("--maxnreg", choices=(64, 80, 96, 112, 128), type=int,
                        help="explicit register ceiling; only for candidates supporting it")
    parser.add_argument("--output")
    parser.add_argument("--cases", default="", help="comma-separated names; default full matrix")
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-cases", default="c1_p128,c8_p2048")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
