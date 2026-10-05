#!/usr/bin/env python3
"""Publish Phase-5 evidence with strict measured-source and numerical binding.

This is a CPU-only stdlib postprocessor. It never runs a model, changes a
threshold, removes slow repetitions, or substitutes profiler throughput for
serving throughput. Failed prototypes remain independent negative evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

EXPECTED_MATRIX = tuple((c, o) for c in (1, 4, 8) for o in (128, 512))
REQUEST_METRICS = ("offered_ttft_s", "submitted_ttft_s", "offered_e2e_s",
                   "submitted_e2e_s", "delivery_tpot_s", "submission_lag_s")
BULK_SAMPLE_FIELDS = {"requests", "nonempty_batch_delivery_times_s",
    "unique_nonempty_batch_event_gaps_s", "verified_request_details"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    path = Path(path).resolve()
    return json.loads(path.read_text()), {"path": str(path), "sha256": file_sha(path)}


def production_map(source, *, nano_only=False):
    """Normalize old qualification, operator, and serving fingerprint schemas."""
    if "source_file_sha256" in source:
        result = {key: value for key, value in source["source_file_sha256"].items()
                  if key.startswith(("nanovllm:", "jetspec:"))}
    else:
        raw = source.get("production_file_sha256", source.get("production_files", {}))
        result = {}
        for key, value in raw.items():
            if key.startswith("nanovllm/"):
                result["nanovllm:" + key[len("nanovllm/"):]] = value
            elif key.startswith("official_jetspec:"):
                result["jetspec:" + key[len("official_jetspec:"):]] = value
    if nano_only:
        result = {key: value for key, value in result.items() if key.startswith("nanovllm:")}
    require(result, "missing nonempty production file map")
    return result


def revision(source):
    return source.get("revision", source.get("head", source.get("git", {}).get("head")))


def clean(source):
    status = source.get("worktree_status", source.get("status_porcelain",
                        source.get("git", {}).get("status_porcelain")))
    return isinstance(status, str) and not status.strip()


def completed(evidence, role):
    require(evidence.get("passed") is True and evidence.get("status") == "complete",
            f"refusing incomplete/failed {role}")


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def compact_sample(sample):
    result = {key: deepcopy(value) for key, value in sample.items() if key not in BULK_SAMPLE_FIELDS}
    if "requests" in sample:
        result["public_request_output_sha256"] = hashlib.sha256(json.dumps(
            [row["token_ids"] for row in sample["requests"]], separators=(",", ":")).encode()).hexdigest()
    return result


def compact_worker(worker):
    result = {key: deepcopy(value) for key, value in worker.items() if key not in ("samples", "warmups")}
    for phase in ("samples", "warmups"):
        result[phase] = [compact_sample(sample) for sample in worker[phase]]
    return result


def validate_worker(worker, role):
    completed(worker, role)
    require(worker["source"] == worker["source_end"], f"{role} source changed during benchmark")
    require(clean(worker["source"]) and revision(worker["source"]), f"{role} snapshot is not clean/pinned")
    production_map(worker["source"])
    require(worker["serving_policy"].get("enable_chunked_prefill") is False and
            worker["serving_policy"].get("attention_backend") == "sdpa", f"{role} serving policy changed")
    require(tuple((c["concurrency"], c["output_cap_scale"]) for c in worker["cases"]) == EXPECTED_MATRIX,
            f"{role} does not contain original complete c1/c4/c8 x128/512 matrix")
    path = worker["actual_tree_path"]
    require(path.get("profiler_removed_for_timed_samples") is True, f"{role} path profiler leaked")
    require(any(row["function"] == path["required_original_symbol"] and row["calls"] > 0
                for row in path["calls"]), f"{role} actual requested operator was not witnessed")
    for case in worker["cases"]:
        samples = [sample for sample in worker["samples"] if sample["case_index"] == case["case_index"]]
        warmups = [sample for sample in worker["warmups"] if sample["case_index"] == case["case_index"]]
        require(len(samples) >= 3 and len(warmups) >= 1, f"{role} needs >=1 warmup and >=3 formal samples per case")
        require(len({sample["repeat"] for sample in samples}) == len(samples), f"{role} duplicate repeat IDs")
        for sample in samples:
            require(sample["profiler_active_during_timed"] is False, f"{role} profiler active during formal sample")
            require(sample["workload_sha256"] == case["workload_sha256"], f"{role} per-sample workload changed")
            require((sample["concurrency"], sample["output_cap_scale"]) ==
                    (case["concurrency"], case["output_cap_scale"]), f"{role} sample dimensions changed")
            require(sample["exactly_once_and_cap_passed"] and sample["after_cleanup"]["used_blocks"] == 0,
                    f"{role} delivery/allocator correctness failed")
            require(sample["actual_output_tokens"] > 0 and sample["wall_s"] > 0 and
                    math.isfinite(sample["tokens_per_second"]) and sample["tokens_per_second"] > 0,
                    f"{role} invalid throughput")
            require(math.isclose(sample["tokens_per_second"], sample["actual_output_tokens"] / sample["wall_s"],
                                 rel_tol=1e-12), f"{role} throughput not derived from actual output and wall clock")


def matched_summary(reference, candidate):
    for worker, role in ((reference, "reference"), (candidate, "candidate")):
        validate_worker(worker, role)
    for key in ("manifest_sha256", "manifest_file_sha256", "cases", "config", "serving_policy", "models"):
        require(reference[key] == candidate[key], f"practical serving mismatch: {key}")
    for key in ("shape", "dtype", "bytes", "block_size"):
        require(reference["pool"][key] == candidate["pool"][key], f"KV pool geometry mismatch: {key}")
    for key in ("torch", "cuda", "gpu", "compute_capability", "executable", "packages"):
        require(reference["environment"][key] == candidate["environment"][key], f"environment mismatch: {key}")
    result = []
    for case in reference["cases"]:
        arms = []
        output_counts = []
        for worker in (reference, candidate):
            samples = [sample for sample in worker["samples"] if sample["case_index"] == case["case_index"]]
            output_counts.append({sample["actual_output_tokens"] for sample in samples})
            arms.append({"label": worker["label"],
                "raw_throughput_tok_s": [sample["tokens_per_second"] for sample in samples],
                "median_throughput_tok_s": median(sample["tokens_per_second"] for sample in samples),
                "median_request_metric_p50": {key: median(sample["request_metrics"][key]["p50"] for sample in samples)
                                               for key in REQUEST_METRICS},
                "median_delivery_gap_p95_s": median(sample["per_request_delivery_gap_distribution_s"]["p95"] for sample in samples),
                "worst_delivery_gap_max_s": max(sample["per_request_delivery_gap_distribution_s"]["max"] for sample in samples),
                "peak_gpu_allocated_bytes": max(sample["peak_gpu_allocated_bytes"] for sample in samples),
                "peak_gpu_reserved_bytes": max(sample["peak_gpu_reserved_bytes"] for sample in samples),
                "peak_leased_pages": max(sample["peak_used_pages"] for sample in samples),
                "median_emitted_tokens_per_packed_verify": median(sample["mean_effective_output_block_tokens_per_packed_verify_call"] for sample in samples),
                "median_emitted_tokens_per_verified_request": median(sample["mean_effective_output_block_tokens_per_verified_request"] for sample in samples),
                "allocator_cleanup_all_passed": True})
        require(len(output_counts[0]) == 1 and output_counts[0] == output_counts[1], "output caps/count differ between arms/repeats")
        ratio = arms[1]["median_throughput_tok_s"] / arms[0]["median_throughput_tok_s"]
        result.append({**case, "actual_output_tokens": next(iter(output_counts[0])), "arms": arms,
                       "candidate_over_reference_median_throughput_ratio": ratio,
                       "median_throughput_change_percent": (ratio - 1) * 100})
    return result


def bind_qualification(qualification, candidate):
    completed(qualification, "candidate qualification")
    for field in ("all_gates_passed", "source_unchanged", "allocator_cleanup"):
        require(qualification.get(field) is True, f"qualification missing {field}")
    require(qualification.get("fp32_pre_bf16_scaled_max_and_relative_rms_bound") == 1e-4 and
            qualification.get("full_network_bf16_scaled_max_and_relative_rms_bound") == 2 ** -6,
            "qualification changed existing numerical thresholds")
    require(qualification.get("cross_shape_token_bitwise_required") is False, "unexpected cross-shape token contract")
    require(qualification.get("synthetic") and qualification.get("trained_same_state"), "synthetic-only qualification is insufficient")
    require(production_map(qualification["source"]) == production_map(candidate["source"]),
            "qualification and serving production files differ; rerun final snapshot qualification")
    require(clean(qualification["source"]) and revision(qualification["source"]) == revision(candidate["source"]),
            "qualification is not the measured clean revision")
    for key in ("torch", "cuda", "gpu"):
        require(qualification["environment"][key] == candidate["environment"][key], f"qualification environment differs: {key}")
    paths = qualification["source"]["fingerprint_scope"]["model_paths"]
    for key in ("target", "draft"):
        require(str(Path(paths[key]).resolve()) == candidate["models"][key]["path"], f"qualification {key} checkpoint differs")
    isolation = qualification["packed_request_isolation"]
    require(isolation["real_eight_request_packed_shape"] and isolation["finite_same_shape_controls_passed"], "packed isolation did not pass")
    require(all(qualification["lifecycle"]["semantic_gates"].values()), "EOS/cancel/max-token/dynamic gates did not pass")
    require(qualification["lifecycle"]["fixed_schedule_replay_exact"], "fixed schedule lifecycle replay failed")
    require(all(qualification["allocator_pressure"][key] for key in ("deferred_seen", "recovered")), "allocator pressure recovery failed")
    require(all(qualification["preemption_recompute"][key] for key in
        ("preemption_seen", "resume_seen", "output_exactly_once", "fixed_schedule_replay_exact", "scratch_one_page_or_less")), "preemption/recompute failed")
    require(qualification["chunked_lifecycle_recompute"]["output_exactly_once"], "chunked capability lifecycle failed")
    return {"all_measured_production_files_equal": True, "file_count": len(production_map(candidate["source"])),
            "clean_revision_equal": True, "model_paths_equal": True, "torch_cuda_gpu_equal": True,
            "existing_numerical_thresholds_unchanged": True, "full_trained_and_lifecycle_qualification": True}


def bind_micro(micro, qualification, candidate):
    completed(micro, "selected operator microbenchmark")
    require(micro.get("NOT_A_SERVING_BENCHMARK") is True and micro.get("source_unchanged") is True,
            "operator microbenchmark was mixed with serving or source changed")
    require(production_map(micro["source"], nano_only=True) == production_map(candidate["source"], nano_only=True),
            "selected microbenchmark production differs from final serving snapshot")
    require(clean(micro["source"]) and revision(micro["source"]) == revision(candidate["source"]), "selected micro is not final clean revision")
    measured = micro["candidate"]
    selected = qualification["candidate"]
    require(measured["module"] + ":" + measured["function"] == selected["entry"] and
            measured["file_sha256"] == selected["module_sha256"] and
            measured["function_sha256"] == selected["function_source_sha256"], "micro and qualification selected different operator bytes")
    require(micro["candidate_variant"] == selected["executed_options"],
            "micro/qualification options differ or implicit options were not recorded; pass selected options explicitly")
    for key in ("torch", "cuda", "gpu", "compute_capability", "executable"):
        require(micro["environment"][key] == candidate["environment"][key], f"operator environment differs: {key}")
    require(micro["measurement"]["warmups_per_arm"] >= 1 and micro["measurement"]["repeats"] >= 3,
            "micro warmup/repeat policy incomplete")
    for case in micro["cases"]:
        require(case["validation"]["passed"] is True, "operator numerical/isolation case failed")
        for arm in ("baseline", "candidate"):
            require(len(case["samples"][arm]) >= 3 and case["warmups"][arm], "operator raw repetition matrix incomplete")
    return {"final_production_files_and_clean_revision_equal": True, "operator_source_and_explicit_options_equal": True,
            "environment_equal": True, "must_not_use_micro_speedup_as_serving_speedup": True}


def bind_profile(profile, worker):
    completed(profile, "serving profile")
    require(profile.get("diagnostic_only") is True, "profile not explicitly separated from timed performance")
    require(production_map(profile["source"]) == production_map(worker["source"]) and
            revision(profile["source"]) == revision(worker["source"]), "profile and serving use different production")
    require(profile["source"] == profile["source_end"] and clean(profile["source"]), "profile source changed or dirty")
    for key in ("models", "config", "manifest_sha256", "manifest_file_sha256"):
        require(profile[key] == worker[key], f"profile serving mismatch: {key}")
    require(profile["policy"] == worker["serving_policy"], "profile policy changed")
    for key in ("torch", "cuda", "gpu", "compute_capability", "executable", "packages"):
        require(profile["environment"][key] == worker["environment"][key], f"profile environment differs: {key}")
    result = deepcopy(profile)
    for key in ("warmup", "kernel_window_run", "stage_run"):
        if key in result:
            result[key] = compact_sample(result[key])
    result["must_not_use_for_serving_throughput"] = True
    return result


def named_paths(values):
    result = []
    seen = set()
    for value in values:
        label, separator, path = value.partition("=")
        require(separator and label and path and label not in seen, "named artifact must be unique label=path")
        seen.add(label)
        result.append((label, path))
    return result


def publication(reference, candidate, micro, qualification, *, profiles=(), negatives=()):
    summary = matched_summary(reference, candidate)
    qualification_binding = bind_qualification(qualification, candidate)
    micro_binding = bind_micro(micro, qualification, candidate)
    representative = next(case for case in summary if (case["concurrency"], case["output_cap_scale"]) == (8, 512))
    profile_records = []
    for role, evidence, artifact in profiles:
        require(role in ("reference", "candidate"), "profile role must be reference or candidate")
        profile_records.append({"role": role, "artifact": artifact,
            "evidence": bind_profile(evidence, reference if role == "reference" else candidate)})
    negative_records = [{"label": label, "artifact": artifact, "evidence": deepcopy(evidence),
        "not_promoted_to_selected_qualification": True,
        "note": "Original status/passed/error/raw samples remain unchanged; a fast prototype or passed micro is not full-model qualification."}
        for label, evidence, artifact in negatives]
    return {"schema_version": 1, "phase": 5, "status": "complete", "passed": True,
        "kind": "profile-driven Target verification Tree Attention optimization qualification",
        "comparison": "reference JetSpec vs final candidate JetSpec; same original final manifest and serving policy, not upstream AR comparison",
        "representative_case_preregistered": {"concurrency": 8, "output_cap_scale": 512},
        "representative_median_serving_speedup": representative["candidate_over_reference_median_throughput_ratio"],
        "serving_summary": summary, "reference_worker": compact_worker(reference), "candidate_worker": compact_worker(candidate),
        "operator_microbenchmark": deepcopy(micro), "candidate_qualification": deepcopy(qualification),
        "measured_source_bindings": {"qualification": qualification_binding, "microbenchmark": micro_binding},
        "diagnostic_profiles": profile_records, "negative_results": negative_records,
        "contract": {"fp32_pre_bf16_scaled_max_and_relative_rms_bound": 1e-4,
            "full_network_bf16_scaled_max_and_relative_rms_bound": 2 ** -6,
            "cross_shape_token_bitwise_required": False,
            "finite_same_shape_request_and_branch_isolation_required": True,
            "bitwise_cross_layout_arithmetic_claimed": False,
            "precision_implementation_note": "Use the selected operator source/qualification to distinguish scalar FP32, FP32 GQA reuse, or Tensor Core TF32x3; fixed numerical bounds do not imply identical multiplication/reduction implementation."},
        "publication_note": "All warmups and all formal repetitions retained, including slow outliers. Only bulky delivery/token arrays omitted from public serving copies; complete raw artifacts remain identified by path and SHA256. Profile and operator throughput are never substituted for serving throughput. Mixed output caps/two-wave arrivals are the unchanged original workload, not uniform 128/512 outputs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "micro", "qualification", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--profile", action="append", default=[],
                        help="reference=path or candidate=path; multiple captures use candidate:c8=path etc")
    parser.add_argument("--negative", action="append", default=[], help="unique experiment-label=path; original failures preserved")
    parser.add_argument("--review", help="optional attributed test/profile/workspace notes; does not change any gate")
    args = parser.parse_args()
    evidence, artifacts = {}, {}
    inputs = [args.reference, args.candidate, args.micro, args.qualification]
    inputs += [path for _, path in named_paths(args.profile) + named_paths(args.negative)]
    if args.review:
        inputs.append(args.review)
    require(Path(args.output).resolve() not in {Path(path).resolve() for path in inputs}, "publication must not overwrite raw evidence")
    for name in ("reference", "candidate", "micro", "qualification"):
        evidence[name], artifacts[name] = load(getattr(args, name))
    profiles = []
    for label, path in named_paths(args.profile):
        role = label.split(":", 1)[0]
        profile, artifact = load(path)
        artifact["capture_label"] = label
        profiles.append((role, profile, artifact))
    negatives = [(label, *load(path)) for label, path in named_paths(args.negative)]
    result = publication(**evidence, profiles=profiles, negatives=negatives)
    result["full_raw_artifacts"] = artifacts
    if args.review:
        notes, artifact = load(args.review)
        result["review_notes"] = {"artifact": artifact, "notes": notes,
            "supplemental_only_not_a_substitute_for_bound_qualification": True}
    result["publisher_source"] = {"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__),
                                  "CPU_only_stdlib_postprocessing": True}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(target.resolve()), "passed": result["passed"],
                      "representative_median_serving_speedup": result["representative_median_serving_speedup"]}))


if __name__ == "__main__":
    main()
