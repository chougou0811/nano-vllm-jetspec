#!/usr/bin/env python3
"""Compact completed, matched Phase-4 artifacts; never rerun a GPU workload."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

from jetspec_phase3 import distribution
from jetspec_phase4 import BASELINE, compare


def digest(path):
    with Path(path).open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def load(path):
    path = Path(path).resolve()
    return path, json.loads(path.read_text())


def provenance(path, data):
    keys = ("revision", "repository", "production_source_sha256", "script_sha256",
            "benchmark_helper_sha256", "qualification_harness_source_sha256")
    return {"path": str(path), "sha256": digest(path),
            **{key: data[key] for key in keys if key in data}}


def compact(value):
    """Keep numerical/gate summaries, not streamed tokens or diagnostic traces."""
    omitted = {"token_ids", "steps", "requests", "events", "arrivals", "delivery_batches",
               "run", "runs", "first", "second", "early_c8", "verify_shapes", "rows",
               "production_file_sha256", "feature_updates", "draft_batches"}
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items() if key not in omitted}
    if isinstance(value, list):
        if value and all(type(item) is bool for item in value):
            return {"count": len(value), "all_passed": all(value)}
        return [compact(item) for item in value] if len(value) <= 32 else {"count": len(value)}
    return value


def peak(values):
    captured = [value for value in values if value is not None]
    return max(captured) if captured else None


def summarize(rows):
    steps = [step for row in rows for step in row["steps"]]
    verified = [request for step in steps for request in step["verification_requests"]]
    capacities = [step.get(key, {}) for step in steps for key in ("capacity", "capacity_during_verify")]
    timings = {key: distribution([request[key] for row in rows for request in row["requests"]
                                 if request.get(key) is not None]) for key in rows[0]["metrics"]}
    totals = {row["actual_output_tokens"] for row in rows}
    hashes = {row["workload_sha256"] for row in rows}
    if len(totals) != 1 or len(hashes) != 1:
        raise ValueError("cannot merge repeats with different workloads or output counts")
    result = {"samples": len(rows), "requests": sum(len(row["requests"]) for row in rows),
              "workload_sha256": next(iter(hashes)), "actual_output_tokens_per_sample": next(iter(totals)),
              "tok_s_median": statistics.median(row["tokens_per_second"] for row in rows),
              "tok_s_distribution": distribution([row["tokens_per_second"] for row in rows]),
              "request_latency_s": timings, "verification_rounds": sum(bool(s["node_counts"]) for s in steps),
              "verified_requests": len(verified), "verified_queries": sum(sum(s["node_counts"]) for s in steps),
              "mean_raw_accepted_draft_per_verified_request": statistics.mean(
                  r["accepted_draft_length"] for r in verified) if verified else None,
              "mean_emitted_output_per_verified_request": statistics.mean(
                  r["emitted_output_length"] for r in verified) if verified else None,
              "max_packed_requests": peak(len(s["node_counts"]) for s in steps),
              "all_live_arrival": all(row["dynamic_live_arrival_seen"] for row in rows),
              "cleanup_passed": all(row["capacity_after"].get("requests") == 0 and
                  row["capacity_after"].get("pending_destination_blocks") == 0 for row in rows)}
    for name in ("peak_reserved_kv_slots", "peak_live_kv_slots", "peak_gpu_allocated_bytes"):
        result[name] = peak(row.get(name) for row in rows)
    for name in ("target_feature_reserved_bytes", "target_feature_live_bytes", "draft_cache_bytes"):
        result["peak_observed_" + name] = peak(capacity.get(name) for capacity in capacities)
    return result


def profile_summary(entry, artifact):
    profile, run = entry["profile"], entry["run"]
    steps = run["steps"]
    if entry["kind"] == "kernel_window":
        lengths, window, started = defaultdict(int), [], False
        for step in steps:
            started |= max(lengths.values(), default=0) >= profile.get("threshold_outputs", float("inf"))
            if started and len(window) < profile.get("steps", 0):
                window.append(step)
            for event in step["events"]:
                if event["kind"] == "tokens":
                    lengths[event["request_id"]] += len(event["token_ids"])
        steps = window
    requests = [request for step in steps for request in step["verification_requests"]]
    counts = {"rounds": sum(bool(step["node_counts"]) for step in steps),
              "verified_requests": len(requests), "queries": sum(sum(s["node_counts"]) for s in steps)}
    result = {"artifact": artifact, "kind": entry["kind"], "workload_counts": counts}
    if entry["kind"] == "stage":
        keys = ("stages", "copy_payload_bytes", "feature_append_copy_bytes", "feature_history_copy_bytes",
                "feature_update_copy_bytes", "draft_batched_forward_calls", "draft_serial_forward_calls",
                "peak_reserved_kv_slots", "feature_traffic_note", "timing_note")
        result.update({key: profile[key] for key in keys if key in profile})
        result["mean_raw_accepted_draft_per_verified_request"] = statistics.mean(
            r["accepted_draft_length"] for r in requests) if requests else None
        result["mean_emitted_output_per_verified_request"] = statistics.mean(
            r["emitted_output_length"] for r in requests) if requests else None
        batches = profile.get("draft_batches", [])
        result["max_reported_draft_padding_payload_bytes_per_proposal"] = peak(b.get("transient_kv_padding_bytes") for b in batches)
        result["max_draft_cache_storage_bytes"] = peak(b.get("cache_storage_bytes_after") for b in batches)
    else:
        result.update({key: profile[key] for key in ("captured", "steps", "threshold_outputs",
            "cpu_sync_self_ms", "cpu_launch_self_ms", "all_observed_cuda_device_ms", "device_time_note") if key in profile})
        result["cpu_boundary_records"] = [row for row in profile.get("top_cpu_ops", [])
            if any(term in row["operator"] for term in ("Synchronize", "Memcpy", "LaunchKernel"))]
        result["top_cpu_ops"] = profile.get("top_cpu_ops", [])[:12]
        result["top_cuda_kernels"] = profile.get("top_cuda_kernels", [])[:12]
        trace = Path(profile.get("trace_path", ""))
        if trace.is_file():
            result["trace_artifact"] = {"path": str(trace.resolve()), "sha256": digest(trace)}
    return result


def build_report(run_paths, qualification_path, unit_log=None, unit_exit_code=None):
    paths = [Path(path).resolve() for path in run_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate input artifact would double-count repeats")
    records, excluded = [], []
    for path in paths:
        path, data = load(path)
        if "smoke" in path.name or path.name == "qualification.json":
            excluded.append({**provenance(path, data), "reason": "preliminary; excluded from formal evidence"})
        else:
            records.append((path, data))
    baselines = [(path, data) for path, data in records if data.get("revision", "").startswith(BASELINE)]
    if not baselines:
        raise ValueError("a clean formal 7bcb754 baseline is required")
    baseline_path, baseline = max(baselines, key=lambda item: len(item[1].get("samples", [])))
    qpath, qualification = load(qualification_path)
    if (qualification.get("worktree_status", "").strip() or qualification.get("all_gates_passed") is not True
            or qualification.get("allocator_clean") is not True or qualification.get("source_unchanged") is not True
            or qualification.get("qualification_harness_unchanged") is not True):
        raise ValueError("qualification must be completed, clean, source-stable and passing")
    model_paths = qualification.get("fingerprint_scope", {}).get("model_paths", {})
    if any(str(Path(model_paths.get(key, "")).resolve()) != baseline["models"][key]["resolved_path"] for key in ("target", "draft")):
        raise ValueError("qualification and timed runs must use the same Target/Draft paths")
    groups = {}
    for path, data in records:
        if data.get("kind") != "phase4_matched_jetspec":
            raise ValueError(f"not a formal Phase4 run: {path}")
        is_baseline = data["revision"].startswith(BASELINE)
        expected = baseline["production_source_sha256"] if is_baseline else qualification["production_source_sha256"]
        if data["production_source_sha256"] != expected:
            raise ValueError(f"production fingerprint differs from its qualified revision: {path}")
        shared = set(data.get("summary", {})) & set(baseline.get("summary", {}))
        subset = {**data, "samples": [row for row in data["samples"] if row["case"] in shared],
                  "summary": {case: data["summary"][case] for case in shared}}
        compare(subset, baseline_path)  # Also checks globals when c16 has no shared case.
        flags = data["optimization_flags"]
        if set(flags) != {"lightweight", "batched_draft", "feature_storage"} or any(type(v) is not bool for v in flags.values()):
            raise ValueError("optimization flags must explicitly contain three boolean policies")
        if is_baseline and any(flags.values()):
            raise ValueError("baseline optimization flags must all be false")
        label = "baseline" if is_baseline else "+".join(key for key, enabled in sorted(flags.items()) if enabled) or "optimized-control"
        group_id = label + "-" + expected[:12]
        group = groups.setdefault(group_id, {"optimization_flags": flags, "production_source_sha256": expected,
            "baseline": is_baseline, "artifacts": [], "profiles": [], "pressure": [], "_samples": defaultdict(list)})
        group["artifacts"].append(provenance(path, data))
        for row in data["samples"]:
            group["_samples"][row["case"]].append(row)
        group["profiles"].extend(profile_summary(entry, str(path)) for entry in data.get("profiles", []))
        if data.get("pressure"):
            group["pressure"].append({"artifact": str(path), **compact(data["pressure"])})
    baseline_group = next(group for group in groups.values() if group["baseline"])
    baseline_cases = {case: summarize(rows) for case, rows in baseline_group["_samples"].items()}
    for group in groups.values():
        group["cases"] = {case: summarize(rows) for case, rows in group.pop("_samples").items()}
        for case, summary in group["cases"].items():
            reference = baseline_cases.get(case)
            summary["comparison_status"] = "matched_baseline" if reference else "unmatched_stress_no_speed_ratio"
            if reference and not group["baseline"]:
                if summary["workload_sha256"] != reference["workload_sha256"]:
                    raise ValueError(f"merged workload differs: {case}")
                summary["speed_ratio"] = summary["tok_s_median"] / reference["tok_s_median"]
    for name in ("baseline-smoke.json", "qualification.json"):
        path = baseline_path.parent / name
        if path.exists() and path not in paths and path != qpath:
            excluded.append({**provenance(*load(path)), "reason": "preliminary; excluded from formal evidence"})
    units = None
    if unit_log is not None:
        path = Path(unit_log).resolve()
        log = path.read_text()
        count = re.search(r"Ran (\d+) tests? in ([\d.]+)s", log)
        lines = log.strip().splitlines()
        status = re.fullmatch(r"OK(?: \(skipped=(\d+)\))?", lines[-1] if lines else "")
        if not count or not status or (unit_exit_code is not None and unit_exit_code != 0):
            raise ValueError("unit evidence is incomplete or failed")
        units = {"path": str(path), "sha256": digest(path), "tests": int(count[1]),
                 "body_wall_s": float(count[2]), "skipped": int(status[1] or 0), "status": "OK",
                 "process_exit_code": unit_exit_code,
                 "exit_code_provenance": "caller-supplied process result; not inferred from log"}
    return {"schema_version": 1, "phase": 4, "kind": "compact reproducible qualification summary",
        "summarizer": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
        "matched_config": baseline["config"], "environment": baseline["environment"],
        "pool": baseline["pool"], "models": baseline["models"], "groups": groups,
        "qualification": {"artifact": provenance(qpath, qualification), "summary": compact(qualification)}, "units": units,
        "excluded_artifacts": excluded, "notes": [
            "Only completed clean artifacts enter medians; independent file/repeat samples are pooled, not averaged medians.",
            "Latencies pool request observations; acceptance means pool verified-request records, not percentile summaries.",
            "Missing auxiliary memory observations are null, not zero; boundary peaks omit unobserved transient buffers.",
            "Profiles are separate from timed samples; host/stream stages overlap and must not be added.",
            "Pageable D2H waits can appear under cudaMemcpyAsync; explicit synchronization reduction alone is not total wait reduction.",
            "Checkpoint manifests are not full cryptographic hashes of weight payloads; see model identity notes."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unit-log", help="optional completed unittest log")
    parser.add_argument("--unit-exit-code", type=int, help="actual process result supplied by the caller")
    args = parser.parse_args()
    if args.unit_exit_code is not None and args.unit_log is None:
        parser.error("--unit-exit-code requires --unit-log")
    inputs = [*args.runs, args.qualification, *([args.unit_log] if args.unit_log else [])]
    if Path(args.output).resolve() in {Path(path).resolve() for path in inputs}:
        parser.error("output must not overwrite an input evidence artifact")
    report = build_report(args.runs, args.qualification, args.unit_log, args.unit_exit_code)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output.resolve()), "groups": len(report["groups"])}))


if __name__ == "__main__":
    main()
