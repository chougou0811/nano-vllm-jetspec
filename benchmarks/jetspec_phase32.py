#!/usr/bin/env python3
"""Matched dynamic-arrival serving benchmarks and separate correctness probes.

Timing uses the public engine add/step/cancel API, one resident Target + Draft,
one fixed KV pool, and identical wall-clock offered arrivals in both modes.
Tokens become observable only at synchronous engine.step() return boundaries.
Qualification instead fixes logical arrival/cancel ticks so replay holds actual
batch/GEMM schedules constant. This does not assert arbitrary-shape AR equality.
"""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

from jetspec_phase3 import distribution, save_json
from jetspec_phase31 import identity as phase31_identity

TARGET = "/root/autodl-tmp/models/modelscope-Qwen3-8B"
DRAFT = "/root/autodl-tmp/models/jetspec-qwen3-8b-020a198caefde24a2891ad827cba7fb977ccdc36"
ORACLE = "/root/autodl-tmp/benchmarks/jetspec-phase0/oracle.json"


@dataclass(frozen=True)
class Arrival:
    request_id: str
    prompt_id: str
    prompt: list[int]
    max_tokens: int
    tree_budget: int
    arrival_s: float = 0.0
    arrival_step: int = 0
    cancel_step: int | None = None
    ignore_eos: bool = True


def workload(prompts, concurrency, max_tokens, arrival_ms):
    """Two waves: the second arrives after service has already started."""
    names = ["natural_language", "math_logic", "long_prompt", "long_continuation"]
    budgets = [63, 31, 47, 63]
    caps = [max_tokens, max_tokens, min(17, max_tokens), max(1, max_tokens // 2)]
    return [Arrival(
        f"r{i}", names[i % 4], list(prompts[names[i % 4]]["prompt_token_ids"]),
        caps[i % 4], budgets[i % 4],
        arrival_s=0.0 if i < concurrency else (i - concurrency + 1) * arrival_ms / 1000,
        arrival_step=0 if i < concurrency else i - concurrency + 1,
    ) for i in range(2 * concurrency)]


def qualification_workload(prompts):
    def req(name, prompt, cap, budget, tick, cancel=None):
        return Arrival(name, prompt, list(prompts[prompt]["prompt_token_ids"]),
                       cap, budget, arrival_step=tick, cancel_step=cancel, ignore_eos=False)
    return [req("eos", "fact", 64, 63, 0), req("running-cancel", "natural_language", 96, 31, 0, 3),
            req("queued-cancel", "long_prompt", 64, 47, 0, 1),
            req("one-token", "math_logic", 1, 63, 2),
            req("late", "long_continuation", 17, 47, 4),
            req("refill", "math_logic", 32, 31, 6)]


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {"index": index, "left": a, "right": b}
    return None if len(left) == len(right) else {"index": min(len(left), len(right)),
                                              "left_length": len(left), "right_length": len(right)}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def fingerprint(args):
    result = phase31_identity(args)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["benchmark_helper_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("jetspec_phase3.py", "jetspec_phase31.py")
    }
    return result


def set_mode(engine, mode, draft, concurrency):
    if not engine.is_finished():
        raise RuntimeError("benchmark mode switch requires an idle engine")
    engine.scheduler.max_num_seqs = concurrency
    if mode == "jetspec":
        engine.configure_jetspec(draft, max_admissions_per_step=2)
        engine._jetspec_scheduler.max_num_seqs = concurrency
    else:
        engine.disable_jetspec()


class AllocationObserver(AbstractContextManager):
    """Identical O(1) allocator high-water observation in both timed modes."""
    def __init__(self, engine):
        self.manager = engine.scheduler.block_manager
        self.original = self.manager._allocate_block
        self.peak_blocks = len(self.manager.used_block_ids)

    def __enter__(self):
        def allocate():
            result = self.original()
            self.peak_blocks = max(self.peak_blocks, len(self.manager.used_block_ids))
            return result
        self.manager._allocate_block = allocate
        return self

    def __exit__(self, *exc):
        self.manager._allocate_block = self.original


def capacity(engine, mode):
    manager = engine.scheduler.block_manager
    if mode == "jetspec":
        return json_safe(engine._jetspec_scheduler.runtime.capacity_snapshot())
    seqs = list(engine.scheduler.waiting) + list(engine.scheduler.running)
    live = sum(seq.num_cached_tokens for seq in seqs)
    reserved = len(manager.used_block_ids) * manager.block_size
    return {"waiting_count": len(engine.scheduler.waiting), "running_count": len(engine.scheduler.running),
            "allocator_used_blocks": len(manager.used_block_ids), "live_kv_slots": live,
            "reserved_kv_slots": reserved, "amplification": reserved / live if live else None}


def serving_run(engine, specs, *, mode, clock="wall", label="run", max_wall_s=180):
    """Drive arrivals while requests remain active; never call generate()."""
    import torch
    from nanovllm import SamplingParams
    records = {s.request_id: {"request_id": s.request_id, "prompt_id": s.prompt_id,
        "max_tokens": s.max_tokens, "tree_budget": s.tree_budget, "ignore_eos": s.ignore_eos,
        "scheduled_arrival_s": s.arrival_s, "scheduled_arrival_step": s.arrival_step,
        "token_ids": [], "submitted_s": None, "first_token_s": None, "terminal_s": None,
        "status": "not_arrived"} for s in specs}
    submitted, cancelled, ids = set(), set(), {}
    step_trace, arrivals, acceptances = [], [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tick = 0

    def observe(event, now):
        public_id = ids.get(event.get("request_id"))
        if public_id is None:
            raise AssertionError(f"unrecognized serving event request ID: {event}")
        row = records[public_id]
        kind = event["kind"]
        tokens = [int(v) for v in event.get("token_ids", [])]
        if kind == "tokens":
            row["token_ids"].extend(tokens)
            if tokens and row["first_token_s"] is None:
                row["first_token_s"] = now
        elif kind in ("finished", "cancelled", "error"):
            if row["terminal_s"] is not None:
                if row["token_ids"] != tokens:
                    raise AssertionError("duplicate terminal event changed request tokens")
                return
            if row["token_ids"] != tokens:
                raise AssertionError(f"streamed tokens differ from terminal output: {public_id}")
            row.update(status=kind, terminal_s=now, reason=event.get("reason"), metrics=json_safe(event.get("metrics", {})))

    observer = AllocationObserver(engine)
    with observer:
        while True:
            elapsed = time.perf_counter() - start
            if elapsed > max_wall_s:
                raise RuntimeError(f"serving workload deadline exceeded, last steps: {step_trace[-3:]}")
            for spec in specs:
                if spec.request_id in submitted:
                    continue
                ready = elapsed >= spec.arrival_s if clock == "wall" else tick >= spec.arrival_step
                if not ready:
                    continue
                active = not engine.is_finished()
                kwargs = {"tree_budget": spec.tree_budget, "request_id": f"{label}:{spec.request_id}"} if mode == "jetspec" else {}
                ticket = engine.add_request(spec.prompt, SamplingParams(temperature=0, max_tokens=spec.max_tokens,
                                             ignore_eos=spec.ignore_eos), **kwargs)
                ids[ticket] = spec.request_id
                submitted.add(spec.request_id)
                when = time.perf_counter() - start
                records[spec.request_id].update(submitted_s=when, engine_request_id=ticket, status="waiting")
                arrivals.append({"request_id": spec.request_id, "at_s": when, "at_step": tick,
                                 "service_started": bool(step_trace), "engine_active_before_arrival": active})
            for spec in specs:
                if spec.cancel_step is None or tick < spec.cancel_step or spec.request_id not in submitted or spec.request_id in cancelled:
                    continue
                row = records[spec.request_id]
                if row["terminal_s"] is not None:
                    cancelled.add(spec.request_id)
                    continue
                response = engine.cancel_request(row["engine_request_id"])
                cancelled.add(spec.request_id)
                if mode == "ordinary" and response:
                    observe({"kind": "cancelled", "request_id": row["engine_request_id"], "token_ids": row["token_ids"]},
                            time.perf_counter() - start)
                # JetSpec queues the confirmation for its next delivery boundary.
            if engine.is_finished():
                if len(submitted) == len(specs):
                    break
                if clock == "wall":
                    future = min(s.arrival_s for s in specs if s.request_id not in submitted)
                    time.sleep(min(0.005, max(0.0, future - (time.perf_counter() - start))))
                tick += 1
                continue
            before_step = time.perf_counter()
            completed, num_tokens = engine.step()
            delivered = time.perf_counter() - start
            info = getattr(engine, "last_step_info", None) or {}
            # Both engine modes expose the same authoritative event stream.
            # Large diagnostic terminal results/node argmax lists are excluded
            # from our timing-region record conversion, but not computation.
            events = [{k: v for k, v in event.items() if k != "result"}
                      for event in info.get("events", [])]
            for event in events:
                observe(event, delivered)
            verification = info.get("verification") if mode == "jetspec" else None
            if verification:
                acceptances.extend(verification.get("requests", []))
            recorded_verify = None if verification is None else {
                k: ([{rk: rv for rk, rv in r.items() if rk != "target_argmax_by_node"} for r in v]
                    if k == "requests" else v)
                for k, v in verification.items() if not k.startswith("_")}
            step_trace.append({"step": tick, "returned_s": delivered,
                "wall_s": time.perf_counter() - before_step, "num_tokens": num_tokens,
                "events": json_safe(events), "admitted_ids": json_safe(info.get("admitted_ids", [])),
                "resumed_ids": json_safe(info.get("resumed_ids", [])), "preempted_ids": json_safe(info.get("preempted_ids", [])),
                "capacity": json_safe({**capacity(engine, mode), **(info.get("capacity") or {})}),
                "verification": json_safe(recorded_verify)})
            tick += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    ordered = [records[s.request_id] for s in specs]
    if any(row["terminal_s"] is None for row in ordered):
        raise AssertionError("engine became idle without delivering every terminal record")
    for row in ordered:
        row["ttft_s"] = None if row["first_token_s"] is None else row["first_token_s"] - row["submitted_s"]
        row["e2e_latency_s"] = row["terminal_s"] - row["submitted_s"]
        row["arrival_submission_lag_s"] = None if clock != "wall" else row["submitted_s"] - row["scheduled_arrival_s"]
        row["offered_ttft_s"] = None if clock != "wall" or row["first_token_s"] is None else row["first_token_s"] - row["scheduled_arrival_s"]
        row["offered_e2e_latency_s"] = None if clock != "wall" else row["terminal_s"] - row["scheduled_arrival_s"]
        if len(row["token_ids"]) > row["max_tokens"]:
            raise AssertionError("request output cap was exceeded")
    slots = [step["capacity"].get("reserved_kv_slots", 0) for step in step_trace]
    verify_slots = [(step["verification"] or {}).get("capacity_during_verify", {}).get("reserved_kv_slots", 0)
                    for step in step_trace]
    live = [step["capacity"].get("live_kv_slots", 0) for step in step_trace]
    manager = engine.scheduler.block_manager
    return {"mode": mode, "arrival_clock": clock, "wall_latency_s": wall,
        "actual_output_tokens": sum(len(row["token_ids"]) for row in ordered),
        "tokens_per_second": sum(len(row["token_ids"]) for row in ordered) / wall,
        "ttft_s": distribution([row["ttft_s"] for row in ordered if row["ttft_s"] is not None]),
        "e2e_latency_s": distribution([row["e2e_latency_s"] for row in ordered]),
        "offered_ttft_s": distribution([row["offered_ttft_s"] for row in ordered if row["offered_ttft_s"] is not None]),
        "offered_e2e_latency_s": distribution([row["offered_e2e_latency_s"] for row in ordered if row["offered_e2e_latency_s"] is not None]),
        "arrival_submission_lag_s": distribution([row["arrival_submission_lag_s"] for row in ordered if row["arrival_submission_lag_s"] is not None]),
        "requests": ordered, "arrivals": arrivals, "steps": step_trace,
        "dynamic_live_arrival_seen": any(a["service_started"] and a["engine_active_before_arrival"] for a in arrivals),
        "raw_accepted_draft_tokens": distribution([r["accepted_draft_length"] for r in acceptances]),
        "committed_root_inclusive_path": distribution([len(r["committed_path_indices"]) for r in acceptances]),
        "effective_output_per_verified_request": distribution([len(r["output_block"]) for r in acceptances]),
        "peak_boundary_reserved_kv_slots": max(slots, default=0), "peak_boundary_live_kv_slots": max(live, default=0),
        "peak_reserved_kv_slots_including_verify": max(slots + verify_slots, default=0),
        "peak_allocator_used_blocks": observer.peak_blocks,
        "peak_allocator_reserved_kv_slots": observer.peak_blocks * manager.block_size,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocator_used_after": len(manager.used_block_ids), "capacity_after": capacity(engine, mode),
        "sampling": {"temperature": 0},
        "delivery_semantics": "first token and terminal timestamps are observed at synchronous engine.step return"}


class ServingProbe(AbstractContextManager):
    """Separate stage profiler / raw-KV probe, never installed in timed runs."""
    def __init__(self, engine, *, diagnostic=False, frozen=False):
        import torch
        from nanovllm.speculative.jetspec import state as state_module
        self.torch, self.engine, self.state_module = torch, engine, state_module
        self.runner = engine._jetspec_scheduler.runtime
        self.diagnostic = diagnostic
        self.frozen = frozen
        self.patches, self.events, self.raw_checks, self.history_checks = [], {}, [], []
        self.frozen_checks, self.verify_shapes = [], []
        self.peak_blocks = len(engine.scheduler.block_manager.used_block_ids)
        self.copy_bytes = 0

    def patch(self, obj, name, value):
        original = obj.__dict__[name] if isinstance(obj, type) and name in obj.__dict__ else getattr(obj, name)
        self.patches.append((obj, name, original))
        setattr(obj, name, value)

    def measure(self, stage, function, *args, **kwargs):
        start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        try:
            return function(*args, **kwargs)
        finally:
            end.record()
            self.events.setdefault(stage, []).append((start, end, time.perf_counter() - before))

    def __enter__(self):
        torch, state_module, runner = self.torch, self.state_module, self.runner
        manager = self.engine.scheduler.block_manager
        original_allocate = manager._allocate_block
        def allocate():
            block = original_allocate()
            self.peak_blocks = max(self.peak_blocks, len(manager.used_block_ids))
            return block
        self.patch(manager, "_allocate_block", allocate)
        original_create = runner.create_request
        def create(*args, **kwargs):
            request = self.measure("prefill_request", original_create, *args, **kwargs)
            original_propose = request.drafter.propose_logits
            self.patch(request.drafter, "propose_logits", lambda *a, **k: self.measure("draft", original_propose, *a, **k))
            return request
        self.patch(runner, "create_request", create)
        original_verify = runner._verify_batch
        def verify(requests, trees, transaction, metadata):
            if self.diagnostic:
                transaction.arena.kv_pool[:, :, transaction.arena.blocks] = float("nan")
            outputs = self.measure("target_verify", original_verify, requests, trees, transaction, metadata)
            self.verify_shapes.append(list(metadata.node_counts_host))
            if self.diagnostic and not all(bool(torch.isfinite(value).all().item()) for value in outputs):
                raise AssertionError("poisoned scratch leaked into Target output")
            if self.frozen and not self.frozen_checks:
                from jetspec_phase31 import BatchProbe
                # Reuse the read-only full-Q experiment without installing its
                # global wrappers or changing the production acceptance path.
                BatchProbe.frozen_isolation(self, original_verify, requests, trees,
                                            transaction, metadata, outputs)
            return outputs
        self.patch(runner, "_verify_batch", verify)
        original_build = runner.tree_algorithm.build
        self.patch(runner.tree_algorithm, "build", lambda *a, **k: self.measure("tree_build", original_build, *a, **k))
        from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata
        original_metadata = PackedTreeMetadata.build
        self.patch(PackedTreeMetadata, "build", classmethod(
            lambda cls, *a, **k: self.measure("metadata_build", original_metadata, *a, **k)))
        original_admit = state_module.BatchTreeTransaction.admit
        self.patch(state_module.BatchTreeTransaction, "admit", classmethod(
            lambda cls, *a, **k: self.measure("reserve_transaction", original_admit, *a, **k)))
        original_step = self.engine.step
        self.patch(self.engine, "step", lambda *a, **k: self.measure("engine_step_total", original_step, *a, **k))
        import jetspec.tree as tree_module
        original_accept = tree_module.gpu_tree_accept
        self.patch(tree_module, "gpu_tree_accept", lambda *a, **k: self.measure("acceptance", original_accept, *a, **k))
        original_copy = state_module.copy_accepted_kv
        def copy(pool, source, destination, block_size):
            before = pool[:, :, source // block_size, source % block_size].clone() if self.diagnostic else None
            payload = self.measure("accepted_kv_copy", original_copy, pool, source, destination, block_size)
            self.copy_bytes += payload
            if self.diagnostic:
                after = pool[:, :, destination // block_size, destination % block_size]
                exact = torch.equal(before.contiguous().view(torch.uint8), after.contiguous().view(torch.uint8))
                self.raw_checks.append(exact)
                if not exact:
                    raise AssertionError("accepted all-layer raw KV copy changed bytes")
            return payload
        self.patch(state_module, "copy_accepted_kv", copy)
        original_commit = state_module.BatchTreeTransaction.commit
        def commit(transaction, *args, **kwargs):
            previous = []
            if self.diagnostic:
                for state in transaction.states:
                    slots = state.logical_slots.clone()
                    previous.append((slots, state.kv_pool[:, :, slots // state.block_size, slots % state.block_size].clone()))
            result = self.measure("commit_total", original_commit, transaction, *args, **kwargs)
            if self.diagnostic:
                for state, (slots, history), nodes in zip(transaction.states, previous, transaction.node_slots):
                    now = state.kv_pool[:, :, slots // state.block_size, slots % state.block_size]
                    exact = torch.equal(history.contiguous().view(torch.uint8), now.contiguous().view(torch.uint8))
                    isolated = not bool(torch.isin(nodes, state.logical_slots).any().item())
                    self.history_checks.append(exact and isolated)
                    state.assert_round_invariant()
                    if not exact or not isolated:
                        raise AssertionError("history or tree scratch leaked across serving commit")
            return result
        self.patch(state_module.BatchTreeTransaction, "commit", commit)
        return self

    def metrics(self):
        self.torch.cuda.synchronize()
        return {"stages": {name: {"calls": len(events),
            "host_wall_s": distribution([wall for _, _, wall in events]),
            "host_wall_total_s": sum(wall for _, _, wall in events),
            "stream_elapsed_s": distribution([start.elapsed_time(end) / 1000 for start, end, _ in events]),
            "stream_elapsed_total_s": sum(start.elapsed_time(end) / 1000 for start, end, _ in events)}
            for name, events in self.events.items()}, "copy_payload_bytes": self.copy_bytes,
            "peak_used_blocks": self.peak_blocks, "peak_reserved_kv_slots": self.peak_blocks * self.runner.block_size,
            "all_layer_raw_copy_checks": self.raw_checks, "history_and_rejected_checks": self.history_checks,
            "verify_shapes": self.verify_shapes, "frozen_full_query_checks": self.frozen_checks,
            "timing_note": "stage stream elapsed includes host launch gaps; nested stages must not be added together",
            "diagnostic": self.diagnostic}

    def __exit__(self, *exc):
        for obj, name, original in reversed(self.patches):
            setattr(obj, name, original)


def replay_signature(run):
    ids = {r["engine_request_id"]: r["request_id"] for r in run["requests"]}
    return {"requests": [(r["request_id"], r["status"], r["token_ids"]) for r in run["requests"]],
        "steps": [{"step": s["step"],
            "events": [(e["kind"], ids[e["request_id"]], e["token_ids"]) for e in s["events"]],
            "admitted": [ids[v] for v in s["admitted_ids"]],
            "resumed": [ids[v] for v in s["resumed_ids"]],
            "preempted": [ids[v] for v in s["preempted_ids"]],
            "node_counts": (s["verification"] or {}).get("node_counts", []),
            "paths": [row["committed_path_indices"] for row in (s["verification"] or {}).get("requests", [])]}
            for s in run["steps"]]}


def allocator_pressure(engine, prompts):
    """Real held-page backpressure, followed by release and public-API recovery."""
    from nanovllm import SamplingParams
    manager = engine.scheduler.block_manager
    engine._jetspec_scheduler.runtime.release_idle_scratch()
    held = manager.reserve_provisional(max(0, len(manager.free_block_ids) - 2))
    tokens = (prompts["natural_language"]["prompt_token_ids"] * 16)[:255]
    ticket = None
    blocked = []
    try:
        ticket = engine.add_request(tokens, SamplingParams(temperature=0, max_tokens=8),
                                    request_id="pressure", tree_budget=63)
        for _ in range(2):
            engine.step()
            blocked.append(json_safe(engine.last_step_info))
    finally:
        manager.release_provisional(held)
    recovered = []
    for _ in range(100):
        if engine.is_finished():
            break
        engine.step()
        recovered.extend(engine.last_step_info.get("events", []))
    return {"held_blocks": len(held), "free_blocks_while_held": 2,
        "blocked_steps": blocked, "terminal_events": json_safe(recovered),
        "deferred_seen": any(v.get("blocked") or
            ((v.get("waiting_count", 0) > 0 or v.get("preempted_ids")) and not v.get("verification")) for v in blocked),
        "recovered": engine.is_finished() and any(v["kind"] == "finished" and v["request_id"] == ticket for v in recovered),
        "capacity_after": capacity(engine, "jetspec")}


def preemption_qualification(engine, prompts):
    """Three physical pages force a page-boundary suspend/recompute/resume."""
    manager = engine.scheduler.block_manager
    runner = engine._jetspec_scheduler.runtime
    runner.release_idle_scratch()
    specs = [Arrival(f"preempt-{i}", name,
        (prompts[name]["prompt_token_ids"] * 16)[:250], 8, 63)
        for i, name in enumerate(("natural_language", "math_logic"))]
    runs = []
    for repeat in range(2):
        held = manager.reserve_provisional(max(0, len(manager.free_block_ids) - 3))
        try:
            result = serving_run(engine, specs, mode="jetspec", clock="step", label=f"preempt-{repeat}")
            runs.append(result)
        finally:
            manager.release_provisional(held)
            runner.release_idle_scratch()
    preempted = [request_id for step in runs[0]["steps"] for request_id in step["preempted_ids"]]
    resumed = [request_id for step in runs[0]["steps"] for request_id in step["resumed_ids"]]
    return {"usable_pool_pages": 3, "prompt_tokens": 250, "output_cap": 8, "runs": runs,
            "preemption_seen": bool(preempted), "resume_seen": bool(resumed),
            "output_exactly_once": all(row["status"] == "finished" and len(row["token_ids"]) == 8
                                      for run in runs for row in run["requests"]),
            "fixed_schedule_replay_exact": replay_signature(runs[0]) == replay_signature(runs[1]),
            "scratch_one_page_or_less": all(step["capacity"].get("scratch_blocks", 0) <= 1
                                            for run in runs for step in run["steps"]),
            "capacity_after": capacity(engine, "jetspec")}


def eight_request_isolation(engine, prompts, draft):
    """One real c8 packed Q=408 batch, finite same-shape perturbation controls."""
    set_mode(engine, "jetspec", draft, 8)
    engine._jetspec_scheduler.max_admissions_per_step = 8
    specs = [Arrival(f"iso-{i}", spec.prompt_id, spec.prompt, 32, spec.tree_budget)
             for i, spec in enumerate(workload(prompts, 4, 32, 20))]
    with ServingProbe(engine, diagnostic=True, frozen=True) as probe:
        result = serving_run(engine, specs, mode="jetspec", clock="step", label="c8-isolation")
        checks = probe.metrics()
    expected = [63, 31, 47, 63] * 2
    return {"run": result, "checks": checks, "expected_first_shape": expected,
            "real_eight_request_packed_shape": bool(checks["verify_shapes"]) and checks["verify_shapes"][0] == expected,
            "total_query_tokens": sum(expected),
            "finite_same_shape_controls_passed": len(checks["frozen_full_query_checks"]) == 16 and all(
                check.get("request_logits_bitwise_exact", check.get("path_logits_bitwise_exact", False)) and
                check.get("request_taps_bitwise_exact", check.get("path_taps_bitwise_exact", False))
                for check in checks["frozen_full_query_checks"])}


def kernel_profile(engine, prompts, draft, output):
    """Actual CPU/CUDA operation trace, separate from every throughput sample."""
    import torch
    set_mode(engine, "jetspec", draft, 8)
    engine._jetspec_scheduler.max_admissions_per_step = 8
    specs = [Arrival(f"kernel-{i}", spec.prompt_id, spec.prompt, 8, spec.tree_budget)
             for i, spec in enumerate(workload(prompts, 4, 8, 20))]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        result = serving_run(engine, specs, mode="jetspec", clock="step", label="kernel-profile")
    trace = str(Path(output).with_suffix(".trace.json"))
    profile.export_chrome_trace(trace)
    cpu_rows = sorted(({"operator": event.key, "calls": event.count,
                        "self_cpu_ms": event.self_cpu_time_total / 1000,
                        "total_cpu_ms": event.cpu_time_total / 1000}
                       for event in profile.key_averages()), key=lambda row: row["self_cpu_ms"], reverse=True)
    kernels = {}
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            row = kernels.setdefault(event.name, {"kernel": event.name, "calls": 0, "device_ms": 0.0})
            row["calls"] += 1
            row["device_ms"] += event.time_range.elapsed_us() / 1000
    return {"trace_path": trace, "run": result, "top_cpu_ops": cpu_rows[:20],
            "top_cuda_kernels": sorted(kernels.values(), key=lambda row: row["device_ms"], reverse=True)[:20],
            "note": "profiler-enabled run is diagnostic only; kernel durations are observed device events, not stage stream gaps"}


def run(args):
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    from nanovllm import LLM
    torch.manual_seed(0)
    source = fingerprint(args)
    prompts = {p["id"]: p for p in json.loads(Path(args.oracle).read_text())["prompts"]}
    concurrencies = [int(v) for v in args.concurrencies.split(",")]
    config = {"tensor_parallel_size": 1, "enforce_eager": True, "gpu_memory_utilization": 0.8,
        "max_num_batched_tokens": 4096, "max_model_len": 4096, "max_num_seqs": max(concurrencies), "kvcache_block_size": 256}
    engine = LLM(args.target, **config)
    engine.configure_jetspec(args.draft, max_admissions_per_step=2)
    pool = engine.model_runner.kv_cache
    report = {"schema_version": 1, "kind": "phase32_dynamic_serving", **source,
        "config": {**config, "warmup": args.warmup, "repeats": args.repeats, "max_tokens": args.max_tokens,
                   "arrival_interval_ms": args.arrival_ms, "temperature": 0},
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()},
        "global_pool": {"blocks": int(pool.shape[2]), "slots": int(pool.shape[2] * pool.shape[3]), "bytes": pool.numel() * pool.element_size()},
        "notes": ["Target and Draft remain resident in both matched modes; initialization is excluded",
            "same wall-clock offered load does not imply identical actual batching/arrival-boundary schedule",
            "timed output caps are controlled with ignore_eos=True; EOS/cancel are separately qualified",
            "between-step arrivals are delivered at synchronous API boundaries, not a background ingress thread",
            "ordinary and JetSpec differ numerically under the documented shape-dependent BF16 contract",
            "throughput includes synchronous host arrival/event bookkeeping and identical O(1) allocator observation in both modes",
            "offered-clock TTFT/e2e include between-step ingress lag; submitted-clock latencies are reported separately",
            "profile/poison/raw-history probes are separate from all throughput samples"],
        "samples": [], "qualification": None, "profiles": []}
    save_json(args.output, report)
    try:
        for concurrency in concurrencies:
            specs = workload(prompts, concurrency, args.max_tokens, args.arrival_ms)
            for warmup in range(args.warmup):
                for mode in ("ordinary", "jetspec"):
                    set_mode(engine, mode, args.draft, concurrency)
                    serving_run(engine, specs, mode=mode, label=f"w{concurrency}-{warmup}-{mode}")
            for repeat in range(args.repeats):
                for mode in ("ordinary", "jetspec"):
                    set_mode(engine, mode, args.draft, concurrency)
                    sample = serving_run(engine, specs, mode=mode, label=f"c{concurrency}-{repeat}-{mode}")
                    sample.update(concurrency=concurrency, repeat=repeat)
                    report["samples"].append(sample)
                    save_json(args.output, report)
                    print(f"{mode} c{concurrency} repeat{repeat} actual={sample['actual_output_tokens']} tok/s={sample['tokens_per_second']:.2f} TTFTp50={sample['ttft_s']['p50']}", flush=True)
        if not args.skip_qualification:
            set_mode(engine, "jetspec", args.draft, 2)
            specs = qualification_workload(prompts)
            first = serving_run(engine, specs, mode="jetspec", clock="step", label="qual-a")
            with ServingProbe(engine, diagnostic=True) as probe:
                second = serving_run(engine, specs, mode="jetspec", clock="step", label="qual-b")
                checks = probe.metrics()
            eos = next(r for r in second["requests"] if r["request_id"] == "eos")
            queued = next(r for r in second["requests"] if r["request_id"] == "queued-cancel")
            active = next(r for r in second["requests"] if r["request_id"] == "running-cancel")
            pressure = allocator_pressure(engine, prompts)
            preemption = preemption_qualification(engine, prompts)
            isolation = eight_request_isolation(engine, prompts, args.draft)
            report["qualification"] = {"first": first, "poison_replay": second,
                "fixed_step_schedule_and_tokens_exact": replay_signature(first) == replay_signature(second),
                "eos_finished_before_cap": eos["status"] == "finished" and len(eos["token_ids"]) < eos["max_tokens"],
                "queued_cancel_has_no_tokens": queued["status"] == "cancelled" and not queued["token_ids"],
                "running_cancel_had_tokens": active["status"] == "cancelled" and bool(active["token_ids"]),
                "raw_kv": checks, "allocator_pressure": pressure,
                "preemption": preemption, "c8_isolation": isolation}
            save_json(args.output, report)
        if not args.skip_profile:
            concurrency = max(concurrencies)
            set_mode(engine, "jetspec", args.draft, concurrency)
            with ServingProbe(engine) as probe:
                profile_run = serving_run(engine, workload(prompts, concurrency, min(args.max_tokens, 32), args.arrival_ms),
                                          mode="jetspec", label="profile")
                metrics = probe.metrics()
            report["profiles"].append({"concurrency": concurrency, "run": profile_run, "profile": metrics})
            report["kernel_profile"] = kernel_profile(engine, prompts, args.draft, args.output)
        performance = {}
        for concurrency in concurrencies:
            modes = {}
            for mode in ("ordinary", "jetspec"):
                rows = [s for s in report["samples"] if s["concurrency"] == concurrency and s["mode"] == mode]
                modes[mode] = {"samples": len(rows), "aggregate_tok_s_median": statistics.median(s["tokens_per_second"] for s in rows),
                    "wall_latency_s": distribution([s["wall_latency_s"] for s in rows]),
                    "ttft_s": distribution([r["ttft_s"] for s in rows for r in s["requests"] if r["ttft_s"] is not None]),
                    "e2e_latency_s": distribution([r["e2e_latency_s"] for s in rows for r in s["requests"]]),
                    "offered_ttft_s": distribution([r["offered_ttft_s"] for s in rows for r in s["requests"] if r["offered_ttft_s"] is not None]),
                    "offered_e2e_latency_s": distribution([r["offered_e2e_latency_s"] for s in rows for r in s["requests"] if r["offered_e2e_latency_s"] is not None]),
                    "arrival_submission_lag_s": distribution([r["arrival_submission_lag_s"] for s in rows for r in s["requests"] if r["arrival_submission_lag_s"] is not None]),
                    "peak_boundary_reserved_kv_slots": max(s["peak_boundary_reserved_kv_slots"] for s in rows),
                    "peak_allocator_reserved_kv_slots": max(s["peak_allocator_reserved_kv_slots"] for s in rows),
                    "dynamic_live_arrival_all_runs": all(s["dynamic_live_arrival_seen"] for s in rows)}
            modes["jetspec_vs_ordinary_tok_s_ratio"] = modes["jetspec"]["aggregate_tok_s_median"] / modes["ordinary"]["aggregate_tok_s_median"]
            comparisons = []
            for repeat in range(args.repeats):
                ordinary = next(s for s in report["samples"] if s["concurrency"] == concurrency and s["mode"] == "ordinary" and s["repeat"] == repeat)
                jetspec = next(s for s in report["samples"] if s["concurrency"] == concurrency and s["mode"] == "jetspec" and s["repeat"] == repeat)
                comparisons.extend({"repeat": repeat, "request_id": a["request_id"],
                    "exact": a["token_ids"] == b["token_ids"], "first_divergence": first_difference(a["token_ids"], b["token_ids"])}
                    for a, b in zip(jetspec["requests"], ordinary["requests"]))
            modes["different_schedule_numeric_comparisons"] = comparisons
            performance[f"c{concurrency}"] = modes
        report["summary"] = {"performance": performance,
            "all_timed_requests_finished": all(r["status"] == "finished" for s in report["samples"] for r in s["requests"]),
            "dynamic_live_arrival_all_samples": all(s["dynamic_live_arrival_seen"] for s in report["samples"]),
            "qualification_passed": None if report["qualification"] is None else all(report["qualification"][key] for key in (
                "fixed_step_schedule_and_tokens_exact", "eos_finished_before_cap", "queued_cancel_has_no_tokens", "running_cancel_had_tokens"))
                and bool(report["qualification"]["raw_kv"]["all_layer_raw_copy_checks"])
                and all(report["qualification"]["raw_kv"]["all_layer_raw_copy_checks"])
                and bool(report["qualification"]["raw_kv"]["history_and_rejected_checks"])
                and all(report["qualification"]["raw_kv"]["history_and_rejected_checks"])
                and report["qualification"]["allocator_pressure"]["deferred_seen"]
                and report["qualification"]["allocator_pressure"]["recovered"]
                and all(report["qualification"]["preemption"][key] for key in (
                    "preemption_seen", "resume_seen", "output_exactly_once", "fixed_schedule_replay_exact", "scratch_one_page_or_less"))
                and report["qualification"]["c8_isolation"]["real_eight_request_packed_shape"]
                and report["qualification"]["c8_isolation"]["finite_same_shape_controls_passed"]}
    finally:
        scheduler = getattr(engine, "_jetspec_scheduler", None)
        if scheduler is not None:
            for request_id in list(scheduler.requests):
                engine.cancel_request(request_id)
            for _ in range(100):
                if engine.is_finished():
                    break
                engine.step()
        engine.disable_jetspec()
        report["allocator_clean_after_disable"] = not engine.scheduler.block_manager.used_block_ids
        after = fingerprint(args)
        report["source_fingerprint_after"] = after
        report["source_fingerprint_unchanged"] = all(source[key] == after[key] for key in (
            "production_source_sha256", "script_sha256", "benchmark_helper_sha256"))
        save_json(args.output, report)
    if not report["source_fingerprint_unchanged"]:
        raise AssertionError("source changed while serving worker was running")
    if not report["allocator_clean_after_disable"]:
        raise AssertionError("serving cleanup leaked allocator pages")
    print(json.dumps(report["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--draft", default=DRAFT)
    parser.add_argument("--oracle", default=ORACLE)
    parser.add_argument("--concurrencies", default="1,2,4,8")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--arrival-ms", type=float, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-qualification", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.arrival_ms <= 0 or args.repeats < 1 or args.warmup < 0:
        parser.error("positive output/arrival/repeats and nonnegative warmups required")
    if any(c < 1 for c in [int(v) for v in args.concurrencies.split(",")]):
        parser.error("concurrency must be positive")
    run(args)


if __name__ == "__main__":
    main()
