#!/usr/bin/env python3
"""CPU-only attribution of a saved Tree serving Kineto trace.

Raw profile/trace are never rewritten. CUDA kernels are attributed using their
External id to the originating CPU operation, or (for Triton) correlation id to
the cuda_driver launch inside a named phase range. Exclusive kernel partitions
are distinct from nested CUDA-event stage timeline/wall times.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attribute_trace(trace, *, hidden_size=4096, intermediate_size=12288):
    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
    phases = [e for e in events if e.get("cat") == "user_annotation"
              and e.get("name", "").startswith("phase5.")]
    origins = {e["args"]["External id"]: e for e in events
               if e.get("cat") in ("cpu_op", "user_annotation")
               and "External id" in e.get("args", {})}
    apis = {e["args"]["correlation"]: e for e in events
            if e.get("cat") in ("cuda_runtime", "cuda_driver")
            and "correlation" in e.get("args", {})}
    stages, pieces, shape_evidence = {}, {}, defaultdict(Counter)
    kernel_ms = 0.0
    kernel_count = 0
    for event in events:
        if event.get("cat") != "kernel":
            continue
        origin = origins.get(event.get("args", {}).get("External id"))
        if origin is None:
            origin = apis.get(event.get("args", {}).get("correlation"))
        ranges = {p["name"] for p in phases if origin is not None
                  and (p["pid"], p["tid"]) == (origin["pid"], origin["tid"])
                  and p["ts"] <= origin["ts"] <= p["ts"] + p.get("dur", 0)}
        stage = next((label for name, label in (
            ("phase5.target.verify", "target_verify"), ("phase5.draft", "draft"),
            ("phase5.tree_build", "tree_build"), ("phase5.accept", "accept"))
            if name in ranges), "prefill_and_other_lifecycle")
        if "gemm" in event["name"].lower():
            if "phase5.target.o_proj" in ranges:
                piece = "o_proj_gemm"
            elif "phase5.target.lm_head" in ranges:
                piece = "lm_head_gemm"
            elif "phase5.target.attention_block" in ranges:
                piece = "qkv_gemm"
            elif "phase5.target.mlp" in ranges:
                piece = "mlp_gemm_labelled"
            else:
                shapes = origin.get("args", {}).get("Input Dims", []) if origin else []
                weight_shape = shapes[1] if len(shapes) >= 2 else []
                is_mlp = weight_shape in ([hidden_size, intermediate_size],
                                          [intermediate_size, hidden_size])
                piece = "mlp_gemm_shape_confirmed" if is_mlp else "other_gemm"
            if origin is not None:
                shape_evidence[stage + ":" + piece][json.dumps(
                    origin.get("args", {}).get("Input Dims", []))] += 1
        elif "paged_tree" in event["name"] or "tree_attention" in event["name"] or "packed_tree" in event["name"]:
            piece = "tree_attention"
        elif "phase5.target.reference_qk_norm" in ranges:
            piece = "reference_rms_norm_all"
        elif "phase5.target.attention_block" in ranges:
            piece = "rope_slot_and_other_attention_small_ops_proxy"
        else:
            piece = "other_small_ops"
        duration = event.get("dur", 0) / 1000
        kernel_ms += duration
        kernel_count += 1
        for bucket, key in ((stages, stage), (pieces, stage + ":" + piece)):
            row = bucket.setdefault(key, {"cuda_kernel_count": 0, "gpu_kernel_ms": 0.0})
            row["cuda_kernel_count"] += 1
            row["gpu_kernel_ms"] += duration
    launch_counts, launch_us, sync_counts, sync_us = Counter(), Counter(), Counter(), Counter()
    for event in events:
        if event.get("cat") not in ("cuda_runtime", "cuda_driver"):
            continue
        name = event["name"]
        if "LaunchKernel" in name:
            launch_counts[name] += 1
            launch_us[name] += event.get("dur", 0)
        if "Synchronize" in name:
            sync_counts[name] += 1
            sync_us[name] += event.get("dur", 0)
    tree_ms = sum(v["gpu_kernel_ms"] for k, v in pieces.items() if k.endswith(":tree_attention"))
    verify_ms = stages.get("target_verify", {}).get("gpu_kernel_ms", 0)
    return {"cuda_kernel_count": kernel_count, "summed_cuda_kernel_ms": kernel_ms,
        "exclusive_stages": stages, "exclusive_stage_pieces": pieces,
        "gemm_input_shapes": {key: [{"input_dims": json.loads(shape), "calls": count}
                                    for shape, count in shapes.items()]
                              for key, shapes in shape_evidence.items()},
        "launch_calls": dict(launch_counts), "launch_count": sum(launch_counts.values()),
        "launch_host_api_ms": {key: us / 1000 for key, us in launch_us.items()},
        "synchronization_calls": dict(sync_counts), "synchronization_count": sum(sync_counts.values()),
        "synchronization_host_api_ms": {key: us / 1000 for key, us in sync_us.items()},
        "tree_attention_gpu_ms": tree_ms,
        "tree_attention_share_all_cuda_kernels": tree_ms / kernel_ms if kernel_ms else None,
        "tree_attention_share_target_verify_cuda_kernels": tree_ms / verify_ms if verify_ms else None,
        "prefill_observed": any("flash_fwd_kernel" in e.get("name", "") for e in events
                                if e.get("cat") == "kernel"),
        "notes": ["Exclusive sums count CUDA kernel busy duration only, not end-to-end wall time or timeline idle gaps.",
            "The serving window may include dynamic admission/prefill between verification rounds; the all-kernel denominator includes it.",
            "reference_rms_norm_all includes all reference RMSNorm, not just Q/K norm, despite the historical range label.",
            "RoPE/slot attention small-ops bucket is a proxy, not exclusive isolated RoPE attribution.",
            "CUDA driver and runtime launches are both counted; cuLaunchKernelEx includes the Triton tree launches.",
            "GEMM shapes confirm MLP where forward_dense bypassed the original forward-only module label."]}


def publish(profile_path, output):
    raw = json.loads(Path(profile_path).read_text())
    window = raw["kernel_window"]
    trace_path = window["trace_path"]
    if file_sha(trace_path) != window["trace_sha256"]:
        raise AssertionError("raw trace changed after capture")
    report = attribute_trace(json.loads(Path(trace_path).read_text()))
    if abs(report["summed_cuda_kernel_ms"] - window["summed_cuda_kernel_duration_ms"]) > 1e-6:
        raise AssertionError("CUDA duration partition differs from raw profile")
    if report["cuda_kernel_count"] != window["cuda_kernel_count"]:
        raise AssertionError("kernel count differs from raw profile")
    result = {"schema_version": 1, "diagnostic_only": True,
        "raw_profile_path": str(Path(profile_path).resolve()), "raw_profile_sha256": file_sha(profile_path),
        "raw_trace_path": trace_path, "raw_trace_sha256": file_sha(trace_path),
        "raw_capture_script_sha256": raw["script_sha256"],
        "postprocessor_path": str(Path(__file__).resolve()), "postprocessor_sha256": file_sha(__file__),
        "production_source": raw["source"], "environment": raw["environment"],
        "case": raw["case"], "captured_verify_rounds": window["captured_verify_rounds"],
        "attribution": report}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = publish(args.profile, args.output)
    a = result["attribution"]
    print(json.dumps({key: a[key] for key in ("cuda_kernel_count", "summed_cuda_kernel_ms",
        "tree_attention_share_all_cuda_kernels", "tree_attention_share_target_verify_cuda_kernels",
        "launch_count", "prefill_observed")}, indent=2))


if __name__ == "__main__":
    main()
