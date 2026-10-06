#!/usr/bin/env python3
"""Publish existing measurements and provenance; launches no GPU experiments."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jetspec_official_port_benchmark as benchmark


OFFICIAL_HEAD = "2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f"
OFFICIAL_ROOT = f"https://github.com/hao-ai-lab/JetSpec/blob/{OFFICIAL_HEAD}"


def evidence(path):
    return {"path": str(Path(path).resolve()), "sha256": benchmark.upstream.file_sha(path),
            "evidence": json.loads(Path(path).read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream-results", "old-jetspec-results", "jetspec-results", "flash-validation",
                 "qualification", "gemm-negative", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--profile")
    args = parser.parse_args()
    benchmark.combine(SimpleNamespace(**vars(args), allow_subset_for_diagnostics=False))
    report = json.loads(Path(args.output).read_text())
    qualified, rejected = evidence(args.qualification), evidence(args.gemm_negative)
    benchmark.require(qualified["evidence"]["passed"] and
                      qualified["evidence"]["source"] == qualified["evidence"]["source_end"],
                      "same-state qualification missing or source changed")
    benchmark.require(not rejected["evidence"]["passed"] and
                      rejected["evidence"]["status"] == "failed", "negative witness is not a failure")
    report["qualification"] = qualified
    report["qualification_source_scope"] = {
        "measured_candidate": report["workers"][-1]["source"],
        "qualified_source": qualified["evidence"]["source"],
        "note": "Qualification used 5908d4a; selected bc0df24 removed inactive RMS integration. "
                "Do not assert byte-identical production sources. Graph/RoPE arithmetic and selected policy unchanged.",
        "no_new_exhaustive_tests": True}
    report["negative_results"] = [{"label": "combined QKV and gate/up projection prototype",
        "selected": False, "scope": "first same-state full-network sanity round, not a throughput run",
        **rejected}, {"label": "reference-order Triton RMSNorm prototype", "selected": False,
        "scope": "strided native BF16 kernel witness, not a serving benchmark",
        "scaled_pre_store_error": 0.00019736842105263157, "fixed_bound": 0.0001,
        "max_native_error": 0.001953125, "seed": 53, "shape": [32, 47, 128],
        "strides": [256, 8192, 2], "weight_stride": 2,
        "production_policy": "not exposed or integrated into serving"}]
    report["interpretation"]["concurrency"] = (
        "c1/c4/c8 denote identical max_num_seqs user configuration, not identical strict resident-request caps. "
        "Pristine upstream limits each scheduled batch but admits waiting prefills independently of total running; "
        "measured peak running requests are 2/8/16. JetSpec uses strict active admission. "
        "Both natural policies remain unchanged; native_peak_running_requests is retained in every raw upstream sample.")
    report["official_comparison"] = {
        "sources": {"README": f"{OFFICIAL_ROOT}/README.md",
                    "reference_driver": f"{OFFICIAL_ROOT}/bench/reference/benchmark.py",
                    "engine_driver": f"{OFFICIAL_ROOT}/bench/engine/tps_walltime.py",
                    "graphs": f"{OFFICIAL_ROOT}/jetspec/inference_engine/graph_capture.py",
                    "fused_gemms": f"{OFFICIAL_ROOT}/jetspec/inference_engine/compiled_verify_stack.py"},
        "headline_9_64x": "H100 offline MATH-500/Qwen3-8B/greedy/budget256; reference AR is raw HF, not pristine nano-vLLM",
        "headline_1150_tps": "B200 batch1 optimized engine, graphed Draft/verify, warm steady-state; not our serving workload",
        "verify_only_speedup": "Excludes Draft, prefill, and graph setup; not end-to-end throughput",
        "selected_changes_vs_frozen_phase4": ["qualified page-contiguous prefix specialization",
            "BF16-rounding-preserving RoPE/provisional KV scatter", "exact-row bounded Target CUDA Graph replay"],
        "not_implemented_here": ["Draft CUDA Graph", "full compiled verification stack", "BF16-P Tensor Core attention"],
        "baseline_fairness": "Explicitly eager FlashAttention upstream vs graphed JetSpec Target: practical system comparison; "
            "not a claim against upstream with its own CUDA Graph enabled"}
    if args.profile:
        profile = evidence(args.profile)
        benchmark.require(profile["evidence"]["passed"] and profile["evidence"]["diagnostic_only"],
                          "profile must be complete and explicitly diagnostic")
        report["diagnostic_profile"] = profile
    report["publisher_sha256"] = benchmark.upstream.file_sha(__file__)
    benchmark.save(args.output, report)


if __name__ == "__main__":
    main()
