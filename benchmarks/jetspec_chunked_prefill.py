#!/usr/bin/env python3
"""Portable, opt-in trained-model chunked prefill/recompute qualification.

Requires local --target/--draft checkpoints; no private oracle, GPU downloads,
or absolute environment paths. CPU self-tests: --mode self-test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def distribution(values):
    values = sorted(values)
    def percentile(p):
        if not values:
            return None
        pos = (len(values) - 1) * p
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)
    return {"count": len(values), "p50": percentile(.5), "p95": percentile(.95),
            "max": max(values) if values else None}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def save(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def identity(nano, jetspec):
    hashes = {}
    for label, module in (("nanovllm", nano), ("jetspec", jetspec)):
        root = Path(module.__file__).resolve().parent
        hashes.update({label + ":" + str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(root.rglob("*.py"))})
    return {"nano_source": str(Path(nano.__file__).resolve()), "jetspec_source": str(Path(jetspec.__file__).resolve()),
        "production_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def git_provenance(module):
    root = Path(module.__file__).resolve().parent.parent
    def query(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"checkout": query("rev-parse", "--show-toplevel"), "head": query("rev-parse", "HEAD"),
            "status_porcelain": query("status", "--porcelain"), "note": "source fingerprint is authoritative, not HEAD alone"}


def offered_workload(specs):
    return [{"request_id": spec["request_id"], "prompt_length": len(spec["prompt"]),
             "prompt_token_ids_sha256": hashlib.sha256(json.dumps(spec["prompt"]).encode()).hexdigest(),
             "max_tokens": spec["max_tokens"], "tree_budget": spec["tree_budget"],
             "arrival_s": spec.get("arrival_s", 0)} for spec in specs]


def mode_order(chunks, repeat, policy):
    modes = [0, *chunks]
    return list(reversed(modes)) if policy == "alternating" and repeat % 2 else modes


def matched_summary(samples):
    """Same-process/source/pool comparisons; never equate delivery gaps with ITL."""
    groups = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        groups[(sample["concurrency"], sample["output"], sample["workload_sha256"])][sample["chunk_size"]].append(sample)
    result = []
    for (concurrency, output, workload), modes in sorted(groups.items()):
        require(0 in modes, "matched throughput comparison has no unchunked baseline")
        tokens = {sample["actual_output_tokens"] for runs in modes.values() for sample in runs}
        require(len(tokens) == 1, "matched modes emitted different requested output counts")
        reference = statistics.median(s["tokens_per_second"] for s in modes[0])
        variants = []
        for chunk, runs in sorted(modes.items()):
            request_rows = [r for s in runs for r in s["requests"]]
            throughput = distribution([s["tokens_per_second"] for s in runs])
            variants.append({"chunk_size": chunk, "samples": len(runs), "tokens_per_second": throughput,
                "median_speed_ratio_vs_unchunked": throughput["p50"] / reference,
                "pooled_request_metrics": {key: distribution([r[key] for r in request_rows if r[key] is not None])
                    for key in ("submitted_ttft_s", "offered_ttft_s", "submitted_e2e_s", "offered_e2e_s")},
                "pooled_delivery_event_gaps_s": distribution([g for r in request_rows for g in r["delivery_event_gaps_s"]]),
                "peak_reserved_kv_slots": max(s["peak_reserved_kv_slots"] for s in runs),
                "peak_gpu_allocated_bytes": max(s["peak_gpu_allocated_bytes"] for s in runs),
                "prefill_chunk_calls": distribution([s["prefill_chunk_calls"] for s in runs]),
                "recompute_chunks": sum(s["recompute_chunks"] for s in runs),
                "resume_count": sum(s["resume_count"] for s in runs)})
        result.append({"concurrency": concurrency, "output_cap_scale": output, "workload_sha256": workload,
            "actual_output_tokens_per_sample": tokens.pop(), "variants": variants})
    return {"cases": result, "matching": "identical source/environment/model/fixed-pool within one process; identical offered workload hash and actual output count",
            "latency_aggregation": "pooled request observations, not median-of-p95; throughput median across timed samples",
            "interpretation": "offered TTFT/E2E include scheduler submission delay; gaps are inter-delivery-batch, NOT per-token ITL; small sample counts are descriptive"}


def checkpoint(path):
    root = Path(path).resolve()
    files = {}
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".pt"):
            stat = p.stat()
            files[p.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if p.suffix == ".json":
                files[p.name]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"path": str(root), "files": files, "note": "weight shard metadata, not full weight-payload cryptographic hashes"}


def prompt(tokenizer, length, variant=0):
    text = ("Explain how a database maintains consistency during concurrent transactions. ",
            "Solve the arithmetic problem carefully and describe every intermediate step. ",
            "请分析一个长文本推理任务中的依赖关系，并说明结论。 ")[variant % 3]
    ids = tokenizer.encode(text)
    require(bool(ids), "tokenizer produced an empty corpus seed")
    return (ids * ((length + len(ids) - 1) // len(ids)))[:length]


def bitwise_equal(actual, reference):
    import torch
    a, b = actual.detach().cpu().contiguous(), reference.detach().cpu().contiguous()
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


def tensor_metrics(actual, reference, bound):
    import torch
    actual_cpu, reference_cpu = actual.detach().cpu(), reference.detach().cpu()
    require(actual_cpu.shape == reference_cpu.shape and actual_cpu.dtype == reference_cpu.dtype and
            actual_cpu.numel() > 0, "comparison geometry/dtype differs")
    a, b = actual_cpu.double(), reference_cpu.double()
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "nonfinite prefill operands")
    delta = a - b
    maximum = float(delta.abs().max())
    relative = float(delta.square().mean().sqrt()) / max(float(b.square().mean().sqrt()), 1e-30)
    scaled = maximum / max(1., float(b.abs().max()))
    return {"bitwise_equal": bitwise_equal(actual_cpu, reference_cpu), "max_abs": maximum,
            "scaled_max": scaled, "relative_rms": relative, "bound": bound,
            "passed": scaled <= bound and relative <= bound}


def argmax_witness(actual, reference):
    a, b = actual.cpu().float().reshape(-1, actual.shape[-1]), reference.cpu().float().reshape(-1, reference.shape[-1])
    witnesses = []
    for row in range(a.shape[0]):
        winner, other = int(b[row].argmax()), int(a[row].argmax())
        if winner != other:
            delta = float((a[row] - b[row]).abs().max())
            gap = float(b[row, winner] - b[row, other])
            witnesses.append({"row": row, "reference_winner": winner, "actual_winner": other,
                              "reference_gap": gap, "two_linf_delta": 2 * delta, "within_envelope": gap <= 2 * delta})
    return {"rows": a.shape[0], "flips": len(witnesses), "witnesses": witnesses,
            "note": "necessary near-tie inequality, not a correctness proof"}


def close_target_capture(captured, hook):
    require(bool(captured), "Target prefill did not produce a head prediction")
    logits = captured[-1].clone()
    hook.remove()
    return logits


def numeric_diagnostic_evidence(path, source, models, environment, num_layers):
    """A separate FP32 causal-control artifact is required, not an implied pass."""
    path = Path(path).resolve()
    evidence = json.loads(path.read_text())
    helper_sha = hashlib.sha256(Path(__file__).with_name("jetspec_chunked_numeric_diagnostic.py").read_bytes()).hexdigest()
    trace_sha = hashlib.sha256(Path(__file__).with_name("jetspec_layer_trace.py").read_bytes()).hexdigest()
    require(evidence.get("passed") is True and evidence.get("allocator_clean") is True and
            evidence.get("source_unchanged") is True, "required numeric diagnostic did not pass")
    require(evidence["source"]["production_sha256"] == source["production_sha256"] and
            evidence["source"]["script_sha256"] == source["script_sha256"] and
            evidence["diagnostic_sha256"] == helper_sha and evidence["trace_helper"]["sha256"] == trace_sha,
            "numeric diagnostic source/harness/instrumentation fingerprint differs")
    require(evidence["models"] == models, "numeric diagnostic checkpoints differ")
    require(all(evidence["environment"][key] == environment[key] for key in ("torch", "cuda", "gpu")),
            "numeric diagnostic execution environment differs")
    require(evidence["arguments"]["length"] == 33 and evidence["arguments"]["chunk"] == 1,
            "required numeric diagnostic must exercise P33/chunk1")
    require(evidence["same_shape_independent_reference"]["logical_prompt_equal"] and
            all(evidence["same_shape_independent_reference"][key]["bitwise_equal"] for key in ("kv", "taps")) and
            all(evidence["bf16_layer_major_independent_alignment"][key] for key in
                ("full_kv", "chunk_kv", "full_taps", "chunk_taps")), "numeric diagnostic same-shape control failed")
    require(evidence["fp32_control"]["matmul_allow_tf32"] is False and
            evidence["fp32_control"]["sdpa_backend"] == "MATH", "FP32 diagnostic backend contract differs")
    control = evidence["fp32_layer_major"]
    require([layer["layer"] for layer in control["layers"]] == list(range(num_layers)),
            "numeric diagnostic has missing/reordered FP32 layers")
    require(evidence["fp32_control"]["roundoff_bound"] == 2e-4 and
            all(metric["bound"] == 2e-4 for layer in control["layers"] for metric in
                (layer["key"], layer["value"], layer["output_hidden"])) and
            all(control[key]["bound"] == 2e-4 for key in ("final_hidden", "taps", "target_logits")),
            "numeric diagnostic changed the declared FP32 roundoff envelope")
    require(all(metric["passed"] for layer in control["layers"] for metric in
            (layer["key"], layer["value"], layer["output_hidden"])) and
            all(control[key]["passed"] for key in ("final_hidden", "taps", "target_logits")),
            "numeric diagnostic FP32 roundoff control failed")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": evidence["source"], "diagnostic_sha256": helper_sha, "trace_helper": evidence["trace_helper"], "passed": True,
        "layers": len(control["layers"]), "fp32_control": evidence["fp32_control"],
        "fp32_final_hidden": control["final_hidden"], "fp32_taps": control["taps"],
        "fp32_target_logits": control["target_logits"],
        "note": "separate mandatory causal/numeric evidence; pool utilization differs to permit one-layer FP32 casts; not a performance comparison"}


def raw_kv(runtime, slots):
    return runtime.kv_pool[:, :, slots // runtime.block_size, slots % runtime.block_size].detach().cpu()


def configure(engine, args, concurrency, chunk):
    engine.configure_jetspec(args.draft, enable_chunked_prefill=bool(chunk), prefill_chunk_size=chunk or 256,
        max_prefill_tokens=chunk or args.max_model_len, optimization="serving")
    engine._jetspec_scheduler.max_num_seqs = concurrency
    return engine._jetspec_scheduler.runtime


class Ledger:
    """Exactly-once delivered deltas; terminal tokens must equal that history."""
    def __init__(self, specs):
        self.rows = {s["request_id"]: {"token_ids": [], "deliveries": [], "terminal": None,
            "submitted": None, "offered": s.get("arrival_s", 0), "cap": s["max_tokens"]} for s in specs}

    def consume(self, events, now):
        for event in events:
            row = self.rows[event["request_id"]]
            require(row["terminal"] is None, "event after terminal/duplicate terminal")
            if event["kind"] == "tokens":
                row["token_ids"].extend(event["token_ids"])
                if event["token_ids"] and (not row["deliveries"] or row["deliveries"][-1] != now):
                    row["deliveries"].append(now)
            else:
                require(event["kind"] in ("finished", "cancelled", "error"), "unknown event kind")
                require(row["token_ids"] == event["token_ids"], "terminal does not equal streamed history")
                row["terminal"], row["kind"] = now, event["kind"]
                require(event["kind"] != "error", event.get("reason", "terminal request error"))
            require(len(row["token_ids"]) <= row["cap"], "output cap exceeded")

    def summary(self, allow_cancel=False):
        rows = []
        for request_id, row in self.rows.items():
            require(row["terminal"] is not None, "missing terminal")
            require(row["kind"] == "cancelled" and allow_cancel or len(row["token_ids"]) == row["cap"], "output cap not reached")
            deliveries = row["deliveries"]
            rows.append({"request_id": request_id, "status": row["kind"], "tokens": len(row["token_ids"]),
                "submitted_ttft_s": deliveries[0] - row["submitted"] if deliveries else None,
                "offered_ttft_s": deliveries[0] - row["offered"] if deliveries else None,
                "submitted_e2e_s": row["terminal"] - row["submitted"],
                "offered_e2e_s": row["terminal"] - row["offered"],
                "delivery_event_gaps_s": [b - a for a, b in zip(deliveries, deliveries[1:])]})
        return rows


def serve(engine, specs, *, chunk=0, deadline=600, after_step=None, strict=False):
    import torch
    from nanovllm import SamplingParams
    ledger, submitted, steps = Ledger(specs), set(), []
    runtime = engine._jetspec_scheduler.runtime
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    while len(submitted) != len(specs) or not engine.is_finished():
        now = time.perf_counter() - start
        require(now < deadline and len(steps) < 10000, "serving deadline/step limit exceeded")
        for spec in specs:
            if spec["request_id"] not in submitted and now >= spec.get("arrival_s", 0):
                engine.add_request(spec["prompt"], SamplingParams(temperature=0, max_tokens=spec["max_tokens"], ignore_eos=True),
                    request_id=spec["request_id"], tree_budget=spec["tree_budget"])
                submitted.add(spec["request_id"])
                ledger.rows[spec["request_id"]]["submitted"] = time.perf_counter() - start
        if engine.is_finished():
            time.sleep(.001)
            continue
        engine.step()
        delivered = time.perf_counter() - start
        info = engine.last_step_info
        ledger.consume(info["events"], delivered)
        chunks = info.get("prefill_chunks", [])
        require(not chunk or info.get("prefill_tokens", 0) <= chunk, "per-step prefill budget exceeded")
        require(all(0 <= r["start"] < r["end"] <= r["total_tokens"] for r in chunks), "invalid chunk progress")
        verification = info.get("verification") or {}
        steps.append({"prefill_chunks": chunks, "prefill_tokens": info.get("prefill_tokens", 0),
            "prefilling_count": info.get("prefilling_count", 0), "capacity": info["capacity"],
            "capacity_during_verify": verification.get("capacity_during_verify", {}),
            "resumed_ids": info.get("resumed_ids", []), "preempted_ids": info.get("preempted_ids", []),
            "prefill_preempted_ids": info.get("prefill_preempted_ids", []),
            "verified_ids": verification.get("request_ids", []), "blocked": info.get("blocked", False)})
        if strict:
            for request in runtime.requests.values():
                request.state.assert_round_invariant(validate_device=True)
        if after_step:
            after_step(engine, ledger, info)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    rows = ledger.summary(allow_cancel=after_step is not None)
    gaps = [gap for row in rows for gap in row["delivery_event_gaps_s"]]
    capacities = [step[key] for step in steps for key in ("capacity", "capacity_during_verify")]
    tokens = sum(row["tokens"] for row in rows)
    return {"wall_s": wall, "actual_output_tokens": tokens, "tokens_per_second": tokens / wall,
        "requests": rows, "steps": steps, "delivery_event_gaps_s": distribution(gaps),
        "max_request_p95_delivery_event_gap_s": max((distribution(r["delivery_event_gaps_s"])["p95"] or 0 for r in rows), default=0),
        "metrics": {key: distribution([row[key] for row in rows if row[key] is not None])
                    for key in ("submitted_ttft_s", "offered_ttft_s", "submitted_e2e_s", "offered_e2e_s")},
        "prefill_chunk_calls": sum(len(s["prefill_chunks"]) for s in steps),
        "recompute_chunks": sum(r["is_recompute"] for s in steps for r in s["prefill_chunks"]),
        "resume_count": sum(len(s["resumed_ids"]) for s in steps),
        "peak_reserved_kv_slots": max((c.get("reserved_kv_slots", 0) for c in capacities), default=0),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "capacity_after": engine._jetspec_scheduler.capacity_snapshot(),
        "delivery_note": "inter-delivery-batch gaps at synchronous step return; NOT per-token ITL"}


def prefix_qualification(runtime, args, results=None):
    import torch
    from jetspec_chunked_numeric_diagnostic import chronological_forward
    results = [] if results is None else results
    for length, plans in ((33, [[1] * 33]), (257, [[255, 1, 1]]), (515, [[256, 256, 3]]),
                          (273, [[273], [7, 63, 1, 129, 73]])):
        ids = prompt(runtime.tokenizer, length)
        captured = []
        hook = runtime.target.lm_head.register_forward_hook(lambda _m, _a, out: captured.append(out.reshape(-1, out.shape[-1])[-1].detach().cpu()))
        try:
            reference = runtime.create_request(ids, max_new_tokens=8, tree_budget=31, ignore_eos=True)
        finally:
            hook.remove()
        reference_tokens = reference.state.committed.clone()
        reference_kv = raw_kv(runtime, reference.state.logical_slots)
        reference_taps = reference.state.target_hidden.detach().cpu()
        reference_logits = captured[-1]
        reference_draft = reference.drafter.propose_logits(reference_tokens, runtime.tree_depth,
            target_hidden=reference.state.target_hidden).detach().cpu()
        runtime.cancel(reference)
        for plan in plans:
            context = runtime.begin_prefill(ids, max_new_tokens=8, tree_budget=31, ignore_eos=True)
            captured = []
            hook = runtime.target.lm_head.register_forward_hook(lambda _m, _a, out: captured.append(out.reshape(-1, out.shape[-1])[-1].detach().cpu()))
            request = None
            try:
                for count in plan:
                    old_slots = context.logical_slots
                    historical = raw_kv(runtime, old_slots) if old_slots.numel() else None
                    old_pages = list(context.owned_blocks)
                    request = runtime.prefill_step(context, count)
                    if historical is not None:
                        require(bitwise_equal(historical, raw_kv(runtime, old_slots)), "chunk changed historical KV bytes")
                    owned = request.state.owned_blocks if request else context.owned_blocks
                    require(owned[:len(old_pages)] == old_pages, "canonical prefix pages were replaced")
                require(request is not None and context.promoted, "chunk plan did not complete")
                require(all(r["backend"] == "paged_layerwise_dense_chunk" for r in context.chunk_records), "real model used fake dense fallback")
                require(not context.owned_blocks and context.feature_storage is None, "promotion retained duplicate ownership")
                request.state.assert_round_invariant(validate_device=True)
                actual_target_logits = close_target_capture(captured, hook)
                actual_kv = raw_kv(runtime, request.state.logical_slots)
                dense_hidden, dense_kv, dense_taps = chronological_forward(runtime.target,
                    torch.tensor(ids, dtype=torch.long, device=runtime.kv_pool.device), plan, runtime.target_layer_ids)
                dense_logits = runtime.target.lm_head(dense_hidden[-1:])[-1].detach().cpu()
                independent = {"kv_by_layer": [tensor_metrics(actual_kv[:, i], dense_kv[:, i], 0)
                    for i in range(actual_kv.shape[1])],
                    "taps": tensor_metrics(request.state.target_hidden.squeeze(0), dense_taps, 0),
                    "target_logits": tensor_metrics(actual_target_logits, dense_logits, 0),
                    "logical_prompt_equal": request.state.committed[0, :-1].tolist() == ids}
                kv = [tensor_metrics(actual_kv[:, i], reference_kv[:, i], args.numerical_bound)
                      for i in range(actual_kv.shape[1])]
                taps = tensor_metrics(request.state.target_hidden, reference_taps, args.numerical_bound)
                target = tensor_metrics(actual_target_logits, reference_logits, args.numerical_bound)
                # Same logical anchor: isolate feature drift from an independent near-tie anchor flip.
                draft_logits = request.drafter.propose_logits(reference_tokens, runtime.tree_depth,
                    target_hidden=request.state.target_hidden).detach().cpu()
                draft = tensor_metrics(draft_logits, reference_draft, args.numerical_bound)
                result = {"prompt_length": length, "splits": plan, "kv_by_layer": kv, "taps": taps,
                    "independent_same_shape_reference": independent,
                    "target_logits": target, "forced_same_anchor_draft_logits": draft,
                    "chunk_records": context.chunk_records,
                    "target_argmax": argmax_witness(actual_target_logits, reference_logits),
                    "draft_argmax": argmax_witness(draft_logits, reference_draft),
                    "cross_shape_within_legacy_empirical_envelope": all(m["passed"] for m in [*kv, taps, target, draft]),
                    "earliest_nonbitwise_kv_layer": next((i for i, m in enumerate(kv) if not m["bitwise_equal"]), None)}
                results.append(result)  # Preserve the actual failing witness in the outer report.
                require(independent["logical_prompt_equal"] and all(m["bitwise_equal"] for m in
                    [*independent["kv_by_layer"], independent["taps"], independent["target_logits"]]),
                    "paged differs from independent same-shape chronological reference")
                if len(plan) == 1:
                    require(all(m["bitwise_equal"] for m in [*kv, taps, target, draft]), "same-shape full/chunk control differs")
            finally:
                hook.remove()
                if request is not None and runtime.requests.get(request.request_id) is request:
                    runtime.cancel(request)
                if runtime.prefills.get(context.request_id) is context:
                    runtime.cancel_prefill(context)
    return results


def isolation_qualification(runtime):
    import torch
    ids = prompt(runtime.tokenizer, 33)
    changed = ids[:16] + prompt(runtime.tokenizer, 17, 1)
    contexts = []
    try:
        a, b = [runtime.begin_prefill(p, max_new_tokens=4, ignore_eos=True) for p in (ids, changed)]
        contexts.extend((a, b))
        runtime.prefill_step(a, 16)
        historical = raw_kv(runtime, a.logical_slots)
        runtime.prefill_step(b, 16)
        require(bitwise_equal(historical, raw_kv(runtime, b.logical_slots)), "future suffix changed earlier causal KV")
        require(not set(a.owned_blocks) & set(b.owned_blocks), "partial requests share canonical pages")
        runtime.kv_pool[:, :, b.owned_blocks] = 17  # Finite unrelated prefix perturbation.
        runtime.prefill_step(a, 8)
        actual, taps = raw_kv(runtime, a.logical_slots), a.target_hidden.detach().cpu()
        require(bitwise_equal(historical, actual[:, :, :16]), "unrelated request changed historical KV")
        control = runtime.begin_prefill(ids, max_new_tokens=4, ignore_eos=True)
        contexts.append(control)
        runtime.prefill_step(control, 16)
        runtime.prefill_step(control, 8)
        require(bitwise_equal(actual, raw_kv(runtime, control.logical_slots)) and bitwise_equal(taps, control.target_hidden.cpu()),
                "same-shape partial attention depends on unrelated request pages")
        return {"causal_future_suffix_invariance": True, "disjoint_pages": True,
                "same_shape_unrelated_prefix_perturbation_bitwise_exact": True}
    finally:
        for context in contexts:
            runtime.cancel_prefill(context)


def lifecycle_qualification(engine, args):
    runtime = configure(engine, args, 2, 64)
    mixed = [{"request_id": "resident", "prompt": prompt(engine.tokenizer, 64), "max_tokens": 96, "tree_budget": 3},
             {"request_id": "long", "prompt": prompt(engine.tokenizer, 1024, 1), "max_tokens": 8, "tree_budget": 47, "arrival_s": .02}]
    witnesses = []
    def observe(_engine, ledger, info):
        for record in info.get("prefill_chunks", []):
            if record["request_id"] == "long" and not record["completed"]:
                require(not ledger.rows["long"]["token_ids"], "partial prefill emitted an early anchor")
                if ledger.rows["resident"]["terminal"] is None and ledger.rows["resident"]["token_ids"]:
                    require("resident" in (info.get("verification") or {}).get("request_ids", []), "partial prefill starved resident decode")
                    witnesses.append(record)
    dynamic = serve(engine, mixed, chunk=64, deadline=args.deadline, after_step=observe, strict=True)
    require(witnesses, "no live decode/long partial overlap was witnessed")
    runtime = configure(engine, args, 2, 64)
    cancel = [{"request_id": "cancel-partial", "prompt": prompt(engine.tokenizer, 1024), "max_tokens": 8, "tree_budget": 31}]
    def cancel_partial(_engine, _ledger, info):
        if info.get("prefilling_count") and "cancel-partial" in _engine._jetspec_scheduler.requests:
            _engine.cancel_request("cancel-partial")
    cancelled = serve(engine, cancel, chunk=64, deadline=args.deadline, after_step=cancel_partial, strict=True)
    require(cancelled["actual_output_tokens"] == 0, "cancelled partial emitted output")
    runtime = configure(engine, args, 2, 64)
    manager = runtime.block_manager
    runtime.release_idle_scratch()
    held = manager.reserve_provisional(max(0, len(manager.free_block_ids) - 3))
    pressure = [{"request_id": f"pressure-{i}", "prompt": prompt(engine.tokenizer, 250, i),
                 "max_tokens": 16, "tree_budget": 1} for i in range(2)]
    try:
        recovered = serve(engine, pressure, chunk=64, deadline=args.deadline, strict=True)
        require(recovered["resume_count"] and recovered["recompute_chunks"], "small-budget chunked recompute was not exercised")
        require(set(held).issubset(manager.used_block_ids), "foreign pressure leases were freed")
    finally:
        manager.release_provisional(held)
    return {"live_partial_decode_witnesses": len(witnesses), "dynamic": dynamic, "partial_cancel": cancelled,
            "three_page_pressure_recompute": recovered, "output_exactly_once": True}


def run(args):
    if args.repo:
        sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch, nanovllm, jetspec
    require(torch.cuda.is_available(), "trained-model modes require CUDA")
    if args.repo:
        require(Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()), "nano import escaped --repo")
    torch.manual_seed(0)
    config = dict(tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=args.gpu_memory_utilization,
                  max_num_batched_tokens=4096, max_model_len=args.max_model_len, max_num_seqs=8, kvcache_block_size=256)
    report = {"schema_version": 2, "kind": "chunked prefill/recompute", "source": identity(nanovllm, jetspec),
        "models": {"target": checkpoint(args.target), "draft": checkpoint(args.draft)}, "config": config,
        "git": git_provenance(nanovllm),
        "invocation": {"argv": sys.argv, "arguments": vars(args)},
        "benchmark_policy": {"chunks": args.chunks, "concurrencies": args.concurrencies,
            "output_cap_scales": args.outputs, "long_prompt_lengths": args.prompt_lengths,
            "arrival_ms": args.arrival_ms, "warmup_per_mode_per_case": args.warmup,
            "timed_repeats_per_mode_per_case": args.repeats, "mode_order": args.mode_order,
            "initial_requests_per_case": "concurrency", "later_requests_per_case": "concurrency",
            "initial_prompt_length": 128, "later_arrivals": "(i+1)*arrival_ms after start",
            "output_cap_divisors": [1, 1, 2, 4], "tree_budgets": [63, 31, 47],
            "max_prefill_tokens": "chunk_size, or max_model_len in unchunked mode",
            "warmup_policy": "all mode warmups precede timed rounds for each case",
            "optimization": "serving", "deadline_s": args.deadline},
        "contract": {"cross_shape_token_bitwise_required": False,
            "legacy_empirical_scaled_max_and_relative_rms_bound": args.numerical_bound,
            "legacy_empirical_envelope_is_hard_gate": False,
            "cross_shape_bf16_metrics": "finite metrics and argmax witnesses reported unchanged; the old 2^-6 empirical envelope is not a universal accumulated BF16 bound",
            "hard_gates": ["every chunk plan matches an independent same-shape chronological dense KV/taps/Target-logit reference bitwise",
                "whole-prompt single-chunk control, including initial Draft logits, matches full reference bitwise",
                "historical KV byte immutability, canonical page ownership, causal/request isolation",
                "dynamic admission, partial cancel, chunked pressure/recompute exactly-once output and cleanup",
                "separately recorded same-source trained-model P33/chunk1 FP32 causal-control artifact"],
            "same_shape_controls_bitwise_required": True}, "qualification": {}, "samples": []}
    engine = None
    try:
        engine = nanovllm.LLM(args.target, **config)
        pool = engine.model_runner.kv_cache
        require(pool.dtype == torch.bfloat16, "only BF16 is qualified")
        if args.expected_pool_blocks:
            require(pool.shape[2] == args.expected_pool_blocks, "fixed-pool block count mismatch")
        report["pool"] = {"shape": list(pool.shape), "dtype": str(pool.dtype), "bytes": pool.numel() * pool.element_size()}
        report["environment"] = {"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "dependencies": {name: importlib.metadata.version(name) for name in ("transformers", "triton")}}
        if args.numeric_diagnostic:
            report["required_numeric_diagnostic"] = numeric_diagnostic_evidence(args.numeric_diagnostic,
                report["source"], report["models"], report["environment"], pool.shape[1])
        runtime = configure(engine, args, 2, 64)
        if args.mode in ("qualify", "all"):
            with torch.inference_mode():
                report["qualification"]["prefix"] = []
                prefix_qualification(runtime, args, report["qualification"]["prefix"])
                report["qualification"]["isolation"] = isolation_qualification(runtime)
                report["qualification"]["lifecycle"] = lifecycle_qualification(engine, args)
            report["qualification"]["passed"] = True
            report["qualification"]["cross_shape_within_legacy_empirical_envelope_all_cases"] = all(
                case["cross_shape_within_legacy_empirical_envelope"] for case in report["qualification"]["prefix"])
            save(args.output, report)
        if args.mode in ("benchmark", "all"):
            for concurrency in args.concurrencies:
                for output in args.outputs:
                    specs = [{"request_id": f"r{i}", "prompt": prompt(engine.tokenizer,
                        128 if i < concurrency else args.prompt_lengths[(i-concurrency) % len(args.prompt_lengths)], i),
                        "max_tokens": max(1, output // (1, 1, 2, 4)[i % 4]), "tree_budget": (63, 31, 47)[i % 3],
                        "arrival_s": 0 if i < concurrency else (i-concurrency+1)*args.arrival_ms/1000}
                        for i in range(2 * concurrency)]
                    workload_sha = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()
                    offered = offered_workload(specs)
                    for _ in range(args.warmup):
                        for chunk in [0, *args.chunks]:
                            configure(engine, args, concurrency, chunk)
                            serve(engine, specs, chunk=chunk, deadline=args.deadline)
                            require(engine.model_runner.kv_cache is pool, "pool changed during warmup")
                    for repeat in range(args.repeats):
                        for chunk in mode_order(args.chunks, repeat, args.mode_order):
                            configure(engine, args, concurrency, chunk)
                            result = serve(engine, specs, chunk=chunk, deadline=args.deadline)
                            require(engine.model_runner.kv_cache is pool, "pool changed across matched modes")
                            result.update(concurrency=concurrency, output=output, chunk_size=chunk, repeat=repeat,
                                execution_order=len(report["samples"]), workload_sha256=workload_sha, offered_workload=offered)
                            report["samples"].append(result)
                            report["matched_summary"] = matched_summary(report["samples"])
                            save(args.output, report)
        engine.disable_jetspec()
        report["allocator_clean"] = not engine.scheduler.block_manager.used_block_ids
        report["source_unchanged"] = report["source"] == identity(nanovllm, jetspec)
        require(report["allocator_clean"] and report["source_unchanged"], "cleanup/source stability failed")
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if engine is not None:
            serving = getattr(engine, "_jetspec_scheduler", None)
            if serving is not None:
                for request_id in list(serving.requests):
                    engine.cancel_request(request_id)
                serving.drain_events()
                runtime = serving.runtime
                for context in list(getattr(runtime, "prefills", {}).values()):
                    runtime.cancel_prefill(context)
                for request in list(runtime.requests.values()):
                    runtime.cancel(request)
            engine.exit()
        save(args.output, report)


def self_test():
    import unittest
    class Checks(unittest.TestCase):
        def test_interpolated_distribution(self):
            self.assertAlmostEqual(distribution([0, 1, 2])["p95"], 1.9)
            self.assertIsNone(distribution([])["max"])
        def test_burst_gaps_and_exactly_once(self):
            ledger = Ledger([{"request_id": "r", "max_tokens": 4}])
            ledger.rows["r"]["submitted"] = 0
            ledger.consume([{"kind": "tokens", "request_id": "r", "token_ids": [1]}], .1)
            ledger.consume([{"kind": "tokens", "request_id": "r", "token_ids": [2, 3, 4]},
                            {"kind": "finished", "request_id": "r", "token_ids": [1, 2, 3, 4]}], .3)
            self.assertEqual(ledger.summary()[0]["delivery_event_gaps_s"], [.3-.1])
            with self.assertRaises(AssertionError):
                ledger.consume([{"kind": "finished", "request_id": "r", "token_ids": [1, 2, 3, 4]}], .4)
        def test_cancel_empty_partial(self):
            ledger = Ledger([{"request_id": "r", "max_tokens": 4}])
            ledger.rows["r"]["submitted"] = 0
            ledger.consume([{"kind": "cancelled", "request_id": "r", "token_ids": []}], .1)
            self.assertEqual(ledger.summary(True)[0]["tokens"], 0)
        def test_wrong_terminal_is_rejected(self):
            ledger = Ledger([{"request_id": "r", "max_tokens": 4}])
            with self.assertRaises(AssertionError):
                ledger.consume([{"kind": "finished", "request_id": "r", "token_ids": [9]}], .1)
        def test_alternating_and_fixed_mode_orders(self):
            self.assertEqual(mode_order([64, 256], 0, "alternating"), [0, 64, 256])
            self.assertEqual(mode_order([64, 256], 1, "alternating"), [256, 64, 0])
            self.assertEqual(mode_order([64], 1, "fixed"), [0, 64])
        def test_compact_offered_workload(self):
            specs = [{"request_id": "r", "prompt": [1, 2, 3], "max_tokens": 4, "tree_budget": 7, "arrival_s": .2}]
            offered = offered_workload(specs)[0]
            self.assertEqual(offered["prompt_length"], 3)
            self.assertNotIn("prompt", offered)
            specs[0]["prompt"][0] = 9
            self.assertNotEqual(offered["prompt_token_ids_sha256"], offered_workload(specs)[0]["prompt_token_ids_sha256"])
        def test_matched_medians_and_request_pool(self):
            def sample(chunk, rate, latency):
                return dict(concurrency=1, output=4, workload_sha256="same", chunk_size=chunk,
                    actual_output_tokens=4, tokens_per_second=rate, peak_reserved_kv_slots=256,
                    peak_gpu_allocated_bytes=1024, prefill_chunk_calls=2, recompute_chunks=0, resume_count=0,
                    requests=[dict(submitted_ttft_s=latency, offered_ttft_s=latency,
                        submitted_e2e_s=2*latency, offered_e2e_s=2*latency, delivery_event_gaps_s=[latency])])
            samples = [sample(0, 10, .1), sample(64, 20, .2), sample(0, 30, .3), sample(64, 40, .4)]
            variants = matched_summary(samples)["cases"][0]["variants"]
            self.assertEqual(variants[1]["median_speed_ratio_vs_unchunked"], 1.5)
            self.assertEqual(variants[1]["pooled_request_metrics"]["submitted_ttft_s"]["count"], 2)
            samples[-1]["actual_output_tokens"] = 3
            with self.assertRaises(AssertionError):
                matched_summary(samples)
        def test_unmatched_workload_rejected(self):
            with self.assertRaises(AssertionError):
                matched_summary([dict(concurrency=1, output=4, workload_sha256="other", chunk_size=64)])
    require(unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks)).wasSuccessful(), "CPU checks failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "qualify", "benchmark", "all"), default="qualify")
    parser.add_argument("--target"); parser.add_argument("--draft")
    parser.add_argument("--repo", help="optional selected source checkout, e.g. clean validation tree")
    parser.add_argument("--output", default="chunked-prefill-results.json")
    parser.add_argument("--numeric-diagnostic", help="required passed same-source FP32 diagnostic JSON for qualify/all")
    parser.add_argument("--chunks", type=lambda v: [int(x) for x in v.split(',')], default=[64, 256, 512])
    parser.add_argument("--concurrencies", type=lambda v: [int(x) for x in v.split(',')], default=[1, 4, 8])
    parser.add_argument("--outputs", type=lambda v: [int(x) for x in v.split(',')], default=[128])
    parser.add_argument("--prompt-lengths", type=lambda v: [int(x) for x in v.split(',')], default=[1024, 2048])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.8)
    parser.add_argument("--expected-pool-blocks", type=int)
    parser.add_argument("--numerical-bound", type=float, default=2**-6)
    parser.add_argument("--warmup", type=int, default=1); parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--arrival-ms", type=float, default=20); parser.add_argument("--deadline", type=float, default=600)
    parser.add_argument("--mode-order", choices=("alternating", "fixed"), default="alternating",
                        help="timed mode order reverses on odd repeats (baseline at both ends for repeats=2)")
    args = parser.parse_args()
    if args.mode == "self-test":
        self_test(); return
    if not args.target or not args.draft or not all(Path(p).is_dir() for p in (args.target, args.draft)):
        parser.error("--target and --draft must explicitly select existing local checkpoints")
    if args.mode in ("qualify", "all") and (not args.numeric_diagnostic or not Path(args.numeric_diagnostic).is_file()):
        parser.error("qualify/all requires --numeric-diagnostic JSON from jetspec_chunked_numeric_diagnostic.py")
    if not all(1 <= c <= 8 for c in args.concurrencies) or not all(n > 0 for n in [*args.chunks, *args.outputs, *args.prompt_lengths]):
        parser.error("concurrency must be 1..8 and token counts positive")
    if args.warmup < 0 or args.repeats < 1 or not 0 < args.numerical_bound < 1:
        parser.error("warmup>=0, repeats>=1, numerical bound in (0,1) required")
    if not args.chunks or not args.concurrencies or not args.outputs or not args.prompt_lengths:
        parser.error("benchmark matrices must not be empty")
    if len(set(args.chunks)) != len(args.chunks) or max(args.chunks) > 4096:
        parser.error("chunk sizes must be distinct and <=4096 target batched-token budget")
    if not 0 < args.gpu_memory_utilization <= 1 or args.arrival_ms < 0 or args.deadline <= 0:
        parser.error("GPU utilization in (0,1], arrival>=0, deadline>0 required")
    if args.max_model_len < 1047 or (args.mode in ("benchmark", "all") and
            max(128, *args.prompt_lengths) + max(args.outputs) + 15 > args.max_model_len):
        parser.error("max-model-len must fit qualification and offered benchmark prompt+output+tree lookahead")
    if args.expected_pool_blocks is not None and args.expected_pool_blocks <= 0:
        parser.error("expected-pool-blocks must be positive")
    run(args)


if __name__ == "__main__":
    main()
