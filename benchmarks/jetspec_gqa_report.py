#!/usr/bin/env python3
"""Publish the requested three-case GQA comparison; no GPU experiments or tuning."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jetspec_official_port_benchmark as b


OFFICIAL_HEAD = "2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f"
OFFICIAL_ROOT = f"https://github.com/hao-ai-lab/JetSpec/blob/{OFFICIAL_HEAD}"
EXPECTED_CASES = {(1, 512), (4, 512), (8, 512)}


def evidence(path):
    return {"path": str(Path(path).resolve()), "sha256": b.upstream.file_sha(path),
            "evidence": json.loads(Path(path).read_text())}


def same_tested_modules(kernel, generation, candidate):
    selected = candidate["source"]["source_file_sha256"]
    synthetic = {"nanovllm:" + name.removeprefix("nanovllm/"): sha
                 for name, sha in kernel["source"]["production_files"].items()}
    generated = generation["source"]["source_file_sha256"]
    modules = ("speculative/jetspec/tree_gqa.py", "speculative/jetspec/tree_prefix.py",
               "speculative/jetspec/tree_fusion.py", "speculative/jetspec/paged_backend.py",
               "speculative/jetspec/packed_metadata.py", "speculative/jetspec/target_graph.py",
               "speculative/jetspec/batch_runtime.py", "models/qwen3.py", "engine/llm_engine.py")
    for name in modules:
        key = "nanovllm:" + name
        b.require(synthetic[key] == generated[key] == selected[key],
                  f"tested module differs from measured clean snapshot: {name}")
    changed = [key for key in selected if generated.get(key) != selected[key]]
    b.require(changed == ["nanovllm:speculative/jetspec/runtime.py"],
              "unexpected qualification production differences")
    return {"module_sha256": {name: selected["nanovllm:" + name] for name in modules},
            "whole_source_byte_identical": False,
            "qualification_source_note": "Sanity was run in the development worktree before commit; "
                "tested kernel/serving modules match the clean measured snapshot. Only the user-owned "
                "legacy single-request runtime.py differs; these batch serving probes do not execute that path.",
            "differing_production_files": changed}


def output_agreement(old, candidate):
    rows = []
    for left, right in zip(old["samples"], candidate["samples"]):
        b.require((left["case_index"], left["repeat"]) == (right["case_index"], right["repeat"]),
                  "formal samples cannot be paired for output comparison")
        a = {r["request_id"]: r["token_ids"] for r in left["requests"]}
        c = {r["request_id"]: r["token_ids"] for r in right["requests"]}
        rows.append({"concurrency": left["concurrency"], "output_cap_scale": 512,
            "repeat": left["repeat"], "all_request_tokens_exact": a == c,
            "verified_request_participations_equal": left["verified_request_participations"] ==
                right["verified_request_participations"],
            "output_tokens": left["actual_output_tokens"]})
    return {"samples": rows, "all_exact": all(r["all_request_tokens_exact"] for r in rows),
            "acceptance_participations_unchanged": all(r["verified_request_participations_equal"] for r in rows),
            "scope": "Read-only observation on the nine measured runs, not a universal cross-layout bitwise contract."}


def profile_summary(profile, worker):
    b.require(profile["passed"] and profile["status"] == "complete" and profile["diagnostic_only"],
              "profile is not complete/diagnostic")
    for key in ("production_sha256", "source_file_sha256"):
        b.require(profile["source"][key] == worker["source"][key], "profile production differs from benchmark")
    b.require(profile["source"]["git"]["head"] == worker["source"]["git"]["head"], "profile revision differs")
    b.require(profile["policy"] == worker["execution_policy"], "profile policy differs")
    result = {}
    for row in profile["cases"]:
        key = row["concurrency"], row["output_cap_scale"]
        b.require(key not in result and key in EXPECTED_CASES, "unexpected profile case")
        b.require(row["diagnostic_run"]["exactly_once_and_cap_passed"] and
                  row["diagnostic_run"]["after_cleanup"]["used_blocks"] == 0, "profile delivery/cleanup failed")
        delta = row["graph_delta"]
        b.require(delta["captures"] == 0 and delta["eager_fallbacks"] == 0, "profile did not replay warmed graphs")
        stage = row["stages"]["stages"]["target_verify"]["stream_elapsed_s"]
        window = row["kernel_window"]
        result[key] = {"mean_tree_attention_kernel_ms": window["mean_packed_tree_kernel_ms"],
            "target_verify_event_p50_ms": stage["p50"] * 1000,
            "target_verify_event_p95_ms": stage["p95"] * 1000,
            "attention_kernel_calls_in_window": window["tree_attention_kernel_calls"],
            "attention_gpu_busy_ms_in_window": window["tree_attention_kernel_gpu_ms"],
            "graph_delta": delta}
    b.require(set(result) == EXPECTED_CASES, "profile case matrix incomplete")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream-results", "old-jetspec-results", "jetspec-results", "flash-validation",
                 "kernel-sanity", "generation-sanity", "old-profile", "profile", "output"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    b.combine(SimpleNamespace(**vars(args), allow_subset_for_diagnostics=True))
    report = json.loads(Path(args.output).read_text())
    report.update(status="qualification_pending", passed=False)
    b.save(args.output, report)
    baseline, old, candidate = report["workers"]
    b.require(old["source"]["git"]["head"].startswith("c210e4a"), "wrong current baseline")
    b.require(old["execution_policy"] == {"target_execution": "cuda_graph", "target_kernels": "fused_rope"}
              and candidate["execution_policy"] == {"target_execution": "cuda_graph", "target_kernels": "fused_rope_gqa"},
              "kernel comparison policies changed")
    for worker in report["workers"]:
        b.require({(c["concurrency"], c["output_cap_scale"]) for c in worker["cases"]} == EXPECTED_CASES,
                  "the requested three-case matrix was changed")
        b.require(worker["measurement"]["repeats"] == 3 and
                  worker["measurement"]["warmups_per_case"] == 1, "measurement count changed")
    kernel, generation = evidence(args.kernel_sanity), evidence(args.generation_sanity)
    b.require(kernel["evidence"]["passed"] and generation["evidence"]["passed"], "sanity failed")
    qualification_binding = same_tested_modules(kernel["evidence"], generation["evidence"], candidate)
    old_profile, new_profile = evidence(args.old_profile), evidence(args.profile)
    old_metrics = profile_summary(old_profile["evidence"], old)
    new_metrics = profile_summary(new_profile["evidence"], candidate)
    geometry = lambda window: [{key: row[key] for key in
        ("node_counts", "packed_query_tokens", "prefix_lengths_before_step")}
        for row in window["captured_verify_rounds"]]
    for left, right in zip(old_profile["evidence"]["cases"], new_profile["evidence"]["cases"]):
        b.require(left["concurrency"] == right["concurrency"] and
                  geometry(left["kernel_window"]) == geometry(right["kernel_window"]),
                  "kernel profile windows did not observe identical query/prefix geometry")
    for row in report["summary"]:
        key = row["concurrency"], row["output_cap_scale"]
        left, right = old_metrics[key], new_metrics[key]
        row["diagnostic_profile"] = {"current_c210e4a": left, "new_gqa": right,
            "trained_attention_kernel_busy_time_speedup": left["mean_tree_attention_kernel_ms"] /
                right["mean_tree_attention_kernel_ms"],
            "target_verify_event_p50_speedup": left["target_verify_event_p50_ms"] /
                right["target_verify_event_p50_ms"], "late_window_geometry_equal": True}
    report.update(schema_version=2, status="complete", passed=True,
        benchmark_type="official-capability GQA Target kernel migration",
        diagnostic_subset_not_resume_publication=False,
        publication_contract={"requested_cases": ["c1_o512", "c4_o512", "c8_o512"],
            "warmups_per_case": 1, "formal_samples_per_case": 3, "statistic": "median",
            "immutable_existing_manifest": True, "chunked_prefill": False,
            "no_workload_or_kernel_parameter_search": True,
            "scope_note": "This round explicitly requests three output512 cases, not the previous six-case report."},
        kernel_sanity=kernel, generation_sanity=generation,
        qualification_source_binding=qualification_binding,
        formal_output_observation=output_agreement(old, candidate),
        diagnostic_profiles={"current": old_profile, "candidate": new_profile},
        official_reference={"revision": OFFICIAL_HEAD, "sources": {
            "paged_attention": f"{OFFICIAL_ROOT}/jetspec/inference_engine/paged_tree_attn.py",
            "compile_op": f"{OFFICIAL_ROOT}/jetspec/inference_engine/paged_tree_attn_op.py",
            "compiled_verify": f"{OFFICIAL_ROOT}/jetspec/inference_engine/compiled_verify_stack.py",
            "graph_capture": f"{OFFICIAL_ROOT}/jetspec/inference_engine/graph_capture.py"},
            "ported": ["KV-head-centric GQA/Multi-Q tensor-core tiling", "BF16 probability Tensor Core PV",
                "device ragged Q-block mapping", "causal visible-range loop bound"],
            "preserved": ["FP32 online-softmax running state", "direct canonical Paged KV + round Tree Scratch",
                "ancestor bias/request isolation", "prefix scalar page addressing", "fused RoPE/KV scatter",
                "exact-shape bounded Target CUDA Graph replay", "accepted-only physical commit"],
            "not_ported": ["full compiled verify/custom_op boundary", "Draft CUDA Graph",
                "fused GEMM/norm stack", "true ancestor-aware tile skipping", "B200-specific tile tuning"]},
        publisher_sha256=b.upstream.file_sha(__file__))
    report["interpretation"].update(
        concurrency="c1/c4/c8 are identical configured max_num_seqs. Pristine upstream naturally schedules "
            "waiting prefills independently of its total running count (peak 2/8/16); JetSpec keeps strict admission. "
            "No scheduler was altered to force equality.",
        numerics="Necessary BF16 sanity only: finite/address/isolation/head/ancestor checks and fixed generations. "
            "BF16-P changes arithmetic; observed full-model logit drift is explicitly retained, not called 1e-4-close. "
            "No universal strict token bitwise or model-quality qualification claimed.",
        profiles="GPU kernel busy time is from four late packed verification rounds per case. Whole-run Target event "
            "spans include host launch gaps and profiling overhead. Neither diagnostic throughput nor synthetic "
            "operator event spans replace the nine unprofiled formal samples.",
        ablations="Operator-only one-variable probes; no separate E2E attribution per optimization. GQA/dot/PV "
            "changes are coupled. Visible-range pruning has no measurable gain on the current small-tree fixtures.")
    b.save(args.output, report)


if __name__ == "__main__":
    main()
