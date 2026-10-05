#!/usr/bin/env python3
"""Matched ordinary-AR/JetSpec serving benchmark (raw, unpublished evidence).

Use --repo to select a clean production snapshot. This driver imports the
existing chunked-prefill workload/provenance helpers, but runs no diagnostics.
Only --self-test is CPU-only; the benchmark needs local trained checkpoints.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import AbstractContextManager
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time


HELPER_PATH = Path(__file__).with_name("jetspec_chunked_prefill.py")
_spec = importlib.util.spec_from_file_location("_final_matched_workload_helpers", HELPER_PATH)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
distribution, require, save = _helpers.distribution, _helpers.require, _helpers.save

MODES = ("ordinary_ar", "jetspec")
REQUEST_METRICS = ("offered_ttft_s", "submitted_ttft_s", "offered_e2e_s",
                   "submitted_e2e_s", "delivery_tpot_s", "submission_lag_s")


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_manifest(tokenizer):
    """Exact workload expression used by jetspec_chunked_prefill.py."""
    cases = []
    for concurrency in (1, 4, 8):
        for output in (128, 512):
            specs = [{"request_id": f"r{i}",
                      "prompt": _helpers.prompt(tokenizer, 128 if i < concurrency else
                                                 (1024, 2048)[(i - concurrency) % 2], i),
                      "max_tokens": max(1, output // (1, 1, 2, 4)[i % 4]),
                      "tree_budget": (63, 31, 47)[i % 3],
                      "arrival_s": 0 if i < concurrency else (i - concurrency + 1) * 20 / 1000}
                     for i in range(2 * concurrency)]
            cases.append({"case_index": len(cases), "concurrency": concurrency,
                          "output_cap_scale": output, "specs": specs,
                          "workload_sha256": sha(specs),
                          "offered_workload": _helpers.offered_workload(specs)})
    return {"cases": cases, "manifest_sha256": sha(cases),
            "construction_helper": str(HELPER_PATH.resolve()),
            "helper_sha256": hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest()}


def mode_order(case_index, repeat):
    return list(MODES if (case_index + repeat) % 2 == 0 else reversed(MODES))


def check_legacy_workloads(manifest, path):
    legacy = json.loads(Path(path).read_text())
    expected = defaultdict(set)
    for sample in legacy["samples"]:
        expected[(sample["concurrency"], sample["output"])].add(sample["workload_sha256"])
    checks = []
    for case in manifest["cases"]:
        key = (case["concurrency"], case["output_cap_scale"])
        require(expected[key] == {case["workload_sha256"]}, f"legacy workload mismatch for {key}")
        checks.append({"concurrency": key[0], "output_cap_scale": key[1],
                       "legacy_workload_sha256": case["workload_sha256"], "matched": True})
    return {"path": str(Path(path).resolve()), "artifact_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "cases": checks, "all_six_cases_match": True}


class Ledger:
    """Validate deltas/terminal histories, keeping real burst-delivery times."""
    def __init__(self, specs):
        self.rows = {s["request_id"]: {"request_id": s["request_id"],
                     "prompt_length": len(s["prompt"]), "max_tokens": s["max_tokens"],
                     "tree_budget": s["tree_budget"], "offered_s": s["arrival_s"],
                     "submitted_s": None, "token_ids": [], "deliveries": [], "delivery_events": [],
                     "terminal_s": None, "terminal_count": 0} for s in specs}
        require(len(self.rows) == len(specs), "duplicate manifest request ID")
        self.batch_delivery_times = []

    def submit(self, request_id, now):
        row = self.rows[request_id]
        require(row["submitted_s"] is None, "request submitted twice")
        require(now >= row["offered_s"], "submission predates offered arrival")
        row["submitted_s"] = now

    def consume(self, events, now, actual_to_public):
        nonempty = False
        for event in events:
            require(event["request_id"] in actual_to_public, "unknown actual event request ID")
            row = self.rows[actual_to_public[event["request_id"]]]
            require(row["submitted_s"] is not None, "event before submission")
            require(row["terminal_s"] is None, "duplicate terminal or event after terminal")
            require(now >= row["submitted_s"], "event predates submission")
            tokens = [int(t) for t in event.get("token_ids", [])]
            if event["kind"] == "tokens":
                row["token_ids"].extend(tokens)
                if tokens:
                    row["delivery_events"].append({"at_s": now, "count": len(tokens)})
                    if row["deliveries"] and row["deliveries"][-1]["at_s"] == now:
                        row["deliveries"][-1]["count"] += len(tokens)
                    else:
                        row["deliveries"].append({"at_s": now, "count": len(tokens)})
                    nonempty = True
                require(len(row["token_ids"]) <= row["max_tokens"], "output cap exceeded")
            else:
                require(event["kind"] == "finished", f"unexpected terminal: {event['kind']}")
                require(tokens == row["token_ids"], "terminal history disagrees with exactly-once deltas")
                require(len(tokens) == row["max_tokens"], "ignore-EOS run did not reach cap")
                row["terminal_s"] = now
                row["terminal_count"] += 1
        if nonempty and (not self.batch_delivery_times or self.batch_delivery_times[-1] != now):
            self.batch_delivery_times.append(now)

    def summary(self):
        requests = []
        for original in self.rows.values():
            row = dict(original)
            require(row["terminal_count"] == 1, "request missing exactly one terminal")
            require(len(row["token_ids"]) == row["max_tokens"], "request cap mismatch")
            require(sum(d["count"] for d in row["deliveries"]) == len(row["token_ids"]),
                    "delivery counts disagree with token ledger")
            first, last = row["deliveries"][0]["at_s"], row["deliveries"][-1]["at_s"]
            times = [d["at_s"] for d in row["deliveries"]]
            row["delivery_event_gaps_s"] = [b - a for a, b in zip(times, times[1:])]
            row["delivery_event_gap_distribution_s"] = distribution(row["delivery_event_gaps_s"])
            row.update(offered_ttft_s=first - row["offered_s"],
                       submitted_ttft_s=first - row["submitted_s"],
                       submission_lag_s=row["submitted_s"] - row["offered_s"],
                       offered_e2e_s=row["terminal_s"] - row["offered_s"],
                       submitted_e2e_s=row["terminal_s"] - row["submitted_s"],
                       delivery_tpot_s=(last - first) / (len(row["token_ids"]) - 1)
                       if len(row["token_ids"]) > 1 else None,
                       token_ids_sha256=sha(row["token_ids"]))
            requests.append(row)
        gaps = [b - a for a, b in zip(self.batch_delivery_times, self.batch_delivery_times[1:])]
        return {"requests": requests,
                "actual_output_tokens": sum(len(r["token_ids"]) for r in requests),
                "request_metrics": {key: distribution([r[key] for r in requests if r[key] is not None])
                                    for key in REQUEST_METRICS},
                "nonempty_batch_delivery_times_s": list(self.batch_delivery_times),
                "unique_nonempty_batch_event_gaps_s": gaps,
                "batch_event_gap_distribution_s": distribution(gaps),
                "per_request_delivery_gap_distribution_s": distribution([
                    gap for row in requests for gap in row["delivery_event_gaps_s"]]),
                "exactly_once_and_cap_passed": True}


class AllocationObserver(AbstractContextManager):
    """Same two O(1) host wrappers in both modes, including cached free pages."""
    def __init__(self, manager):
        self.manager = manager
        self.original_block = manager._allocate_block
        self.original_sequence = manager.allocate
        self.peak_blocks = len(manager.used_block_ids)

    def observe(self):
        self.peak_blocks = max(self.peak_blocks, len(self.manager.used_block_ids))

    def __enter__(self):
        def allocate_block(*args, **kwargs):
            result = self.original_block(*args, **kwargs)
            self.observe()
            return result
        def allocate_sequence(*args, **kwargs):
            result = self.original_sequence(*args, **kwargs)
            self.observe()
            return result
        self.manager._allocate_block = allocate_block
        self.manager.allocate = allocate_sequence
        return self

    def __exit__(self, *exc):
        self.manager._allocate_block = self.original_block
        self.manager.allocate = self.original_sequence


def pool_identity(pool):
    return {"object_id": id(pool), "data_ptr": pool.data_ptr(), "shape": list(pool.shape),
            "dtype": str(pool.dtype), "bytes": pool.numel() * pool.element_size()}


def draft_identity(runtime):
    parameters = list(runtime.head.parameters())
    return {"head_object_id": id(runtime.head), "parameter_storage_sha256": sha([
        [p.data_ptr(), list(p.shape), str(p.dtype)] for p in parameters]),
        "parameter_count": sum(p.numel() for p in parameters),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters)}


def idle_boundary(engine, runtime, *, allow_scratch):
    require(engine.is_finished(), "engine is not idle at sample boundary")
    require(not engine.scheduler.waiting and not engine.scheduler.running, "ordinary requests leaked")
    require(not runtime.requests and not runtime.prefills, "JetSpec requests/prefills leaked")
    require(runtime._active_transaction is None and not runtime.arena.active and
            runtime.arena._batch_transaction is None, "scratch transaction/lease still active")
    manager = engine.scheduler.block_manager
    require(runtime.block_manager is manager and runtime.arena.block_manager is manager,
            "cached runtime and scheduler allocator differ")
    expected = set(runtime.arena.blocks) if allow_scratch else set()
    require(manager.used_block_ids == expected, "non-scratch KV pages remain at idle boundary")
    require(len(manager.free_block_ids) + len(expected) == len(manager.blocks), "free page count mismatch")
    require(all(b.ref_count == (1 if b.block_id in expected else 0) for b in manager.blocks),
            "page refcount/ownership mismatch")
    return {"used_blocks": len(manager.used_block_ids), "scratch_blocks": len(runtime.arena.blocks),
            "free_blocks": len(manager.free_block_ids), "hash_entries": len(manager.hash_to_block_id),
            "all_non_scratch_pages_free": True}


def cleanup(engine, runtime):
    before = idle_boundary(engine, runtime, allow_scratch=True)
    engine.disable_jetspec()  # closes serving and releases runner scratch, never Draft weights
    runtime.release_idle_scratch()
    after = idle_boundary(engine, runtime, allow_scratch=False)
    return {"before_cleanup": before, "after_cleanup": after}


def prepare_sample(engine, runtime, draft, concurrency, mode):
    from nanovllm.engine.block_manager import BlockManager
    cleanup(engine, runtime)
    old = engine.scheduler.block_manager
    manager = BlockManager(len(old.blocks), old.block_size)
    engine.scheduler.block_manager = manager
    runtime.block_manager = manager
    runtime.arena.block_manager = manager
    engine.model_runner.config.max_num_seqs = concurrency
    engine.scheduler.max_num_seqs = concurrency
    require(not manager.hash_to_block_id and all(b.hash == -1 and not b.token_ids for b in manager.blocks),
            "new allocator is not cold")
    if mode == "jetspec":
        serving = engine.configure_jetspec(draft, enable_chunked_prefill=False,
            tree_depth=15, tree_width=7, max_tree_budget=63, default_tree_budget=63,
            max_admissions_per_step=2, max_prefill_tokens=4096, optimization="serving")
        require(serving.runtime is runtime and serving.max_num_seqs == concurrency,
                "serving factory changed cached runtime or concurrency")
        require(runtime._lightweight, "timed JetSpec accidentally enables diagnostic events")
    else:
        require(mode == "ordinary_ar" and getattr(engine, "_jetspec_scheduler", None) is None,
                "ordinary mode not selected")
    return idle_boundary(engine, runtime, allow_scratch=False)


def serve(engine, runtime, case, mode, deadline):
    import torch
    from nanovllm import SamplingParams
    ledger = Ledger(case["specs"])
    mapping, submitted = {}, set()
    verify_calls = verified_requests = block_tokens = steps = 0
    by_request = defaultdict(lambda: {"verify_calls": 0, "effective_output_block_tokens": 0})
    # Reset cached allocator memory outside the timer, identically in both
    # modes; compiled kernels and the same Target/Draft weights stay resident.
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident_allocated = torch.cuda.memory_allocated()
    resident_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    with AllocationObserver(engine.scheduler.block_manager) as observer:
        start = time.perf_counter()
        while len(submitted) != len(case["specs"]) or not engine.is_finished():
            now = time.perf_counter() - start
            require(now < deadline, "sample exceeded wall-clock deadline")
            for spec in case["specs"]:
                public = spec["request_id"]
                if public not in submitted and now >= spec["arrival_s"]:
                    options = dict(request_id=public, tree_budget=spec["tree_budget"]) if mode == "jetspec" else {}
                    actual = engine.add_request(spec["prompt"], SamplingParams(
                        temperature=0, max_tokens=spec["max_tokens"], ignore_eos=True), **options)
                    require(actual not in mapping, "actual sequence ID reused within sample")
                    mapping[actual] = public
                    ledger.submit(public, time.perf_counter() - start)
                    submitted.add(public)
            if not engine.is_finished():
                engine.step()
                delivered = time.perf_counter() - start
                info = engine.last_step_info
                require(isinstance(info, dict), "step omitted event report")
                ledger.consume(info.get("events", []), delivered, mapping)
                verification = info.get("verification")
                if verification is not None:
                    require(mode == "jetspec", "ordinary AR unexpectedly used packed verification")
                    require("_verify_events" not in verification and "_commit_events" not in verification,
                            "record_timing enabled in clocked sample")
                    verify_calls += 1
                    for row in verification["requests"]:
                        verified_requests += 1
                        emitted = len(row["output_block"])
                        block_tokens += emitted
                        stats = by_request[row["request_id"]]
                        stats["verify_calls"] += 1
                        stats["effective_output_block_tokens"] += emitted
                steps += 1
            elif len(submitted) != len(case["specs"]):
                time.sleep(.001)  # same idle-arrival waiting policy in both modes
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
    result = ledger.summary()
    result.update(wall_s=wall, tokens_per_second=result["actual_output_tokens"] / wall,
                  steps=steps, packed_verify_calls=verify_calls, verified_request_participations=verified_requests,
                  effective_output_block_tokens=block_tokens,
                  mean_effective_output_block_tokens_per_packed_verify_call=block_tokens / verify_calls
                  if verify_calls else None,
                  mean_effective_output_block_tokens_per_verified_request=block_tokens / verified_requests
                  if verified_requests else None,
                  verified_request_details={k: {**v, "mean_effective_output_block_tokens":
                                               v["effective_output_block_tokens"] / v["verify_calls"]}
                                            for k, v in by_request.items()},
                  peak_used_pages=observer.peak_blocks,
                  peak_reserved_kv_slots=observer.peak_blocks * engine.scheduler.block_manager.block_size,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                  resident_gpu_allocated_bytes=resident_allocated,
                  resident_gpu_reserved_bytes=resident_reserved,
                  peak_gpu_allocated_delta_bytes=torch.cuda.max_memory_allocated() - resident_allocated,
                  peak_gpu_reserved_delta_bytes=torch.cuda.max_memory_reserved() - resident_reserved,
                  gpu_allocated_bytes_at_finish=torch.cuda.memory_allocated(),
                  gpu_reserved_bytes_at_finish=torch.cuda.memory_reserved())
    result.update(cleanup(engine, runtime))
    return result


def matched_summary(samples):
    groups = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        groups[(sample["concurrency"], sample["output_cap_scale"], sample["workload_sha256"])][sample["mode"]].append(sample)
    cases = []
    for (concurrency, output, workload), modes in sorted(groups.items()):
        require(set(modes) == set(MODES), "matched case missing a mode")
        require(all(len(runs) >= 3 for runs in modes.values()), "at least three timed samples per mode required")
        paired = {}
        for mode, runs in modes.items():
            indexed = {s["repeat"]: s for s in runs}
            require(len(indexed) == len(runs), "duplicate timed repeat ID")
            paired[mode] = indexed
        require(set(paired["ordinary_ar"]) == set(paired["jetspec"]), "timed repeat pairing differs")
        totals = {s["actual_output_tokens"] for runs in modes.values() for s in runs}
        require(len(totals) == 1, "modes/repeats emitted unequal real token counts")
        variants = []
        for mode in MODES:
            runs = modes[mode]
            variants.append({"mode": mode, "timed_samples": len(runs),
                "tokens_per_second": distribution([s["tokens_per_second"] for s in runs]),
                "median_wall_s": statistics.median(s["wall_s"] for s in runs),
                "median_of_sample_request_metrics": {key: {stat: statistics.median(
                    s["request_metrics"][key][stat] for s in runs
                    if s["request_metrics"][key][stat] is not None)
                    if any(s["request_metrics"][key][stat] is not None for s in runs) else None
                    for stat in ("p50", "p95", "max")} for key in REQUEST_METRICS},
                "median_of_sample_batch_gap_metrics_s": {stat: statistics.median(
                    s["batch_event_gap_distribution_s"][stat] for s in runs
                    if s["batch_event_gap_distribution_s"][stat] is not None)
                    if any(s["batch_event_gap_distribution_s"][stat] is not None for s in runs) else None
                    for stat in ("p50", "p95", "max")},
                "median_of_sample_per_request_delivery_gap_metrics_s": {stat: statistics.median(
                    s["per_request_delivery_gap_distribution_s"][stat] for s in runs
                    if s["per_request_delivery_gap_distribution_s"][stat] is not None)
                    if any(s["per_request_delivery_gap_distribution_s"][stat] is not None for s in runs) else None
                    for stat in ("p50", "p95", "max")},
                "max_peak_used_pages": max(s["peak_used_pages"] for s in runs),
                "max_peak_reserved_kv_slots": max(s["peak_reserved_kv_slots"] for s in runs),
                "max_peak_gpu_allocated_bytes": max(s["peak_gpu_allocated_bytes"] for s in runs),
                "max_peak_gpu_reserved_bytes": max(s["peak_gpu_reserved_bytes"] for s in runs),
                "max_peak_gpu_allocated_delta_bytes": max(s["peak_gpu_allocated_delta_bytes"] for s in runs),
                "max_peak_gpu_reserved_delta_bytes": max(s["peak_gpu_reserved_delta_bytes"] for s in runs),
                "median_effective_output_block_tokens_per_packed_verify_call": statistics.median(
                    s["mean_effective_output_block_tokens_per_packed_verify_call"] for s in runs)
                    if mode == "jetspec" else None,
                "median_effective_output_block_tokens_per_verified_request": statistics.median(
                    s["mean_effective_output_block_tokens_per_verified_request"] for s in runs)
                    if mode == "jetspec" else None})
        ar, jet = variants
        cases.append({"concurrency": concurrency, "output_cap_scale": output,
                      "workload_sha256": workload, "actual_output_tokens_per_sample": totals.pop(),
                      "jetspec_over_ar_median_throughput_ratio":
                      jet["tokens_per_second"]["p50"] / ar["tokens_per_second"]["p50"],
                      "paired_repeat_throughput_ratios": [{"repeat": repeat,
                          "jetspec_over_ar_ratio": paired["jetspec"][repeat]["tokens_per_second"] /
                          paired["ordinary_ar"][repeat]["tokens_per_second"]}
                          for repeat in sorted(paired["ordinary_ar"])],
                      "median_paired_repeat_throughput_ratio": statistics.median(
                          paired["jetspec"][repeat]["tokens_per_second"] /
                          paired["ordinary_ar"][repeat]["tokens_per_second"]
                          for repeat in paired["ordinary_ar"]), "variants": variants})
    return {"cases": cases,
            "latency_aggregation": "Each sample first computes request p50/p95/max; summary takes the median of the corresponding sample statistic. No pooled-p95 or median-request-p95 substitution.",
            "throughput": "Actual emitted output token count / synchronized sample wall time; ratio of mode medians, not best samples.",
            "delivery_tpot": "(last nonempty delivery time - first nonempty delivery time)/(output token count - 1); this is burst-delivery TPOT, NOT internal per-token ITL.",
            "batch_gaps": "Successive unique step timestamps with at least one nonempty token event, not duplicated across requests. Separate per-request gap statistics coalesce multiple deltas delivered in the same step; raw delivery_events preserve every original delta.",
            "output_blocks": "Sum of trimmed effective verification output_block lengths, excluding initial prefill anchors; denominators are packed forward calls and request-verification participations respectively."}


def source_identity(nano, jetspec):
    return {**_helpers.identity(nano, jetspec),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "git": _helpers.git_provenance(nano), "jetspec_git": _helpers.git_provenance(jetspec)}


def run(args):
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch, nanovllm, jetspec
    from nanovllm.layers import attention
    require(torch.cuda.is_available(), "CUDA is required")
    require(attention.flash_attn_varlen_func is None and attention.flash_attn_with_kvcache is None,
            "expected ordinary SDPA fallback, but FlashAttention callables are active")
    require(Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()), "production import escaped --repo")
    initial_source = source_identity(nanovllm, jetspec)
    require(initial_source["git"]["head"].startswith(args.expected_head), "unexpected production snapshot HEAD")
    require(initial_source["git"]["status_porcelain"] == "", "production snapshot is not clean")
    snapshot_helper = Path(args.repo) / "benchmarks" / HELPER_PATH.name
    require(snapshot_helper.read_bytes() == HELPER_PATH.read_bytes(), "workload helper differs from clean snapshot")
    torch.manual_seed(0)
    config = dict(max_num_seqs=8, max_model_len=4096, max_num_batched_tokens=4096,
                  gpu_memory_utilization=.8, tensor_parallel_size=1, enforce_eager=True, kvcache_block_size=256)
    report = {"schema_version": 1, "status": "raw_unpublished_in_progress", "source": initial_source,
              "invocation": {"argv": sys.argv, "arguments": vars(args)}, "config": config,
              "models": {"target": _helpers.checkpoint(args.target), "draft": _helpers.checkpoint(args.draft)},
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "torch": torch.__version__, "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(), "flash_attn_installed":
                              importlib.util.find_spec("flash_attn") is not None,
                              "packages": {name: importlib.metadata.version(name) for name in
                                           ("transformers", "triton", "numpy")}},
              "baseline": {"label": "ordinary AR serving in the same nano-vLLM fork (SDPA compatibility backend)",
                           "flash_attn_varlen_func_is_none": attention.flash_attn_varlen_func is None,
                           "flash_attn_with_kvcache_is_none": attention.flash_attn_with_kvcache is None,
                           "not_pristine_upstream": True,
                           "existing_fork_compatibility_changes": ["SDPA fallback", "greedy guard", "resident-admission cap"],
                           "snapshot": "b388330", "upstream_reference": "bb823b3"},
              "policy": {"concurrencies": [1, 4, 8], "output_cap_scales": [128, 512],
                         "warmup_per_case_per_mode": args.warmup, "timed_repeats_per_case_per_mode": args.repeats,
                         "warmup_order": "All case/mode warmups finish before any timed sample",
                         "timed_order": "AR/J on even case_index+repeat, J/AR on odd",
                         "enable_chunked_prefill": False, "optimization": "serving", "record_timing": False,
                         "resolved_jetspec_policy": {"tree_depth": 15, "tree_width": 7,
                             "max_tree_budget": 63, "default_tree_budget": 63,
                             "max_admissions_per_step": 2, "max_prefill_tokens": 4096,
                             "max_verify_tokens": 4096, "enable_chunked_prefill": False},
                         "deadline_s": args.deadline,
                         "max_num_seqs": "case concurrency, equal in config and both schedulers",
                         "allocator_start": "Fresh BlockManager with all hashes/cache cleared before every sample",
                         "gpu_allocator_start": "empty_cache plus synchronize outside timer after cold manager reset, with resident allocated/reserved baselines and peak deltas recorded",
                         "draft_weights": "One cached Draft head resident throughout both modes",
                         "cuda_synchronization": "Sample start/end only inside benchmark driver; ordinary production synchronizations unchanged",
                         "resume_example_case_preregistered": {"concurrency": 8, "output_cap_scale": 512},
                         "cross_mode_bitwise_tokens_required": False},
              "warmups": [], "samples": [], "source_frozen_checks": []}
    engine = None
    try:
        engine = nanovllm.LLM(args.target, **config)
        runtime = engine.get_jetspec_batch_runtime(args.draft)
        require(runtime.target is engine.model_runner.model, "JetSpec target is not the ordinary runner model")
        target_parameter_dtypes = sorted({str(p.dtype) for p in engine.model_runner.model.parameters()})
        require(target_parameter_dtypes == ["torch.bfloat16"], "Target parameters are not uniformly BF16")
        report["target_parameter_dtypes"] = target_parameter_dtypes
        report["same_target_model_object"] = True
        pool = engine.model_runner.kv_cache
        require(pool.dtype == torch.bfloat16, "expected BF16 KV pool")
        require(pool.shape[2] == args.expected_pool_blocks, "unexpected fixed KV pool page count")
        report["pool"] = pool_identity(pool)
        report["draft_residency"] = draft_identity(runtime)
        manifest = build_manifest(engine.tokenizer)
        report["legacy_workload_checks"] = check_legacy_workloads(manifest, args.legacy_workload)
        report["manifest_sha256"] = manifest["manifest_sha256"]
        manifest_path = Path(args.output).with_name(Path(args.output).stem + "-manifest.json")
        save(manifest_path, manifest)  # Complete tokens and hashes generated once, before trials.
        report["manifest_file_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        report["manifest_path"] = str(manifest_path.resolve())
        report["cases"] = [{k: v for k, v in case.items() if k != "specs"} for case in manifest["cases"]]
        jobs = [("warmup", case, repeat, mode) for case in manifest["cases"]
                for repeat in range(args.warmup) for mode in mode_order(case["case_index"], repeat)]
        jobs += [("timed", case, repeat, mode) for case in manifest["cases"]
                 for repeat in range(args.repeats) for mode in mode_order(case["case_index"], repeat)]
        save(args.output, report)
        for index, (phase, case, repeat, mode) in enumerate(jobs):
            before_source = source_identity(nanovllm, jetspec)
            require(before_source == initial_source, "source changed before sample")
            cold = prepare_sample(engine, runtime, args.draft, case["concurrency"], mode)
            require(pool_identity(engine.model_runner.kv_cache) == report["pool"] and runtime.kv_pool is pool,
                    "fixed KV pool object/address/shape changed")
            require(draft_identity(runtime) == report["draft_residency"], "Draft weights changed residency")
            print(f"[{index + 1}/{len(jobs)}] {phase} c{case['concurrency']} O{case['output_cap_scale']} {mode} repeat={repeat + 1} START", flush=True)
            sample = serve(engine, runtime, case, mode, args.deadline)
            sample.update(phase=phase, mode=mode, repeat=repeat, case_index=case["case_index"],
                          concurrency=case["concurrency"], output_cap_scale=case["output_cap_scale"],
                          workload_sha256=case["workload_sha256"], cold_allocator=cold,
                          pool=pool_identity(pool), draft_residency=draft_identity(runtime))
            after_source = source_identity(nanovllm, jetspec)
            require(after_source == initial_source, "source changed during sample")
            require(sample["pool"] == report["pool"] and sample["draft_residency"] == report["draft_residency"],
                    "pool or Draft weights changed during sample")
            report["source_frozen_checks"].append({"job": index, "before_after_equal": True,
                                                   "production_sha256": after_source["production_sha256"],
                                                   "driver_sha256": after_source["driver_sha256"]})
            report["warmups" if phase == "warmup" else "samples"].append(sample)
            save(args.output, report)
            print(f"[{index + 1}/{len(jobs)}] DONE tokens={sample['actual_output_tokens']} wall={sample['wall_s']:.3f}s tok/s={sample['tokens_per_second']:.3f} pages={sample['peak_used_pages']}", flush=True)
        report["summary"] = matched_summary(report["samples"])
        require(len(report["samples"]) == 6 * 2 * args.repeats, "timed sample matrix incomplete")
        report["source_end"] = source_identity(nanovllm, jetspec)
        require(report["source_end"] == initial_source, "final frozen-source assertion failed")
        report.update(status="raw_unpublished_complete", passed=True)
        save(args.output, report)
    except BaseException as exception:
        report.update(status="raw_unpublished_failed", passed=False,
                      failure={"type": type(exception).__name__, "message": str(exception)})
        save(args.output, report)
        raise
    finally:
        if engine is not None:
            engine.exit()


def self_test():
    class Tokenizer:
        def encode(self, _):
            return [3, 5, 7]
    manifest = build_manifest(Tokenizer())
    require(len(manifest["cases"]) == 6, "manifest matrix mismatch")
    require(manifest == build_manifest(Tokenizer()), "manifest is not reproducible")
    require(mode_order(0, 0) == ["ordinary_ar", "jetspec"] and
            mode_order(0, 1) == mode_order(1, 0) == ["jetspec", "ordinary_ar"], "unbalanced mode order")
    specs = [{"request_id": "r0", "prompt": [1], "max_tokens": 3, "tree_budget": 1, "arrival_s": 0}]
    ledger = Ledger(specs)
    ledger.submit("r0", .1)
    ledger.consume([{"request_id": 7, "kind": "tokens", "token_ids": [10, 11]}], .2, {7: "r0"})
    ledger.consume([{"request_id": 7, "kind": "tokens", "token_ids": [12]},
                    {"request_id": 7, "kind": "finished", "token_ids": [10, 11, 12]}], .6, {7: "r0"})
    row = ledger.summary()["requests"][0]
    require(abs(row["delivery_tpot_s"] - .2) < 1e-12, "TPOT denominator incorrectly uses burst count")
    require(row["deliveries"] == [{"at_s": .2, "count": 2}, {"at_s": .6, "count": 1}], "burst record changed")
    try:
        ledger.consume([{"request_id": 7, "kind": "finished", "token_ids": [10, 11, 12]}], .7, {7: "r0"})
    except AssertionError:
        pass
    else:
        raise AssertionError("duplicate terminal escaped exactly-once gate")
    print("CPU self-test PASS: workload manifest, balanced order, burst TPOT and exactly-once ledger", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo")
    parser.add_argument("--target")
    parser.add_argument("--draft")
    parser.add_argument("--output", default="final-matched-raw.json")
    parser.add_argument("--expected-head", default="b388330")
    parser.add_argument("--legacy-workload", default=str(HELPER_PATH.with_name("chunked_prefill_benchmark.json")))
    parser.add_argument("--expected-pool-blocks", type=int, default=249)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--deadline", type=float, default=1800)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all(value and Path(value).is_dir() for value in (args.repo, args.target, args.draft)):
        parser.error("--repo, --target and --draft must select existing local directories")
    if args.warmup < 1 or args.repeats < 3 or args.deadline <= 0 or args.expected_pool_blocks <= 0:
        parser.error("warmup>=1, repeats>=3, positive deadline/page count required")
    run(args)


if __name__ == "__main__":
    main()
