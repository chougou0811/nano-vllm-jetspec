#!/usr/bin/env python3
"""Matched Phase-3.2/Phase-4 JetSpec serving and bounded profiling.

The same frozen driver imports either a clean baseline or the optimized tree.
Timing uses synchronous step-return delivery, not background HTTP ingestion.
Profiles, held-page pressure, and correctness probes never enter timed samples.
"""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import jetspec_phase32 as shared
from jetspec_phase3 import distribution, save_json

BASELINE = "7bcb754"


def model_identity(path):
    """Pin location/config/index plus shard metadata without rereading 17GB."""
    root = Path(path).resolve()
    manifest = {}
    for candidate in sorted(root.iterdir()):
        if not candidate.is_file() or candidate.suffix not in (".json",".safetensors",".bin",".pt"):
            continue
        stat = candidate.stat()
        row = {"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}
        if candidate.suffix==".json":
            row["sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        manifest[candidate.name] = row
    return {"resolved_path":str(root),"files":manifest,
        "manifest_sha256":hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        "weight_identity_note":"JSON config/index content SHA256 and shard names/sizes/mtime; full weight payloads are not cryptographically hashed"}


def identity(args):
    result = shared.fingerprint(args)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["benchmark_helper_sha256"]["jetspec_phase32.py"] = hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest()
    return result


def workload(prompts, concurrency, output, arrival_ms, *, short_only=False):
    names = ["natural_language", "long_prompt", "math_logic", "long_continuation"]
    budgets = [63, 31, 47, 63]
    caps = [output, output, max(1, output // 2), max(1, output // 4)]
    specs = []
    for i in range(2 * concurrency):
        name = names[i % 4]
        prompt = list(prompts[name]["prompt_token_ids"])
        if short_only:
            prompt = prompt[:128]
        if i % 4 in (1, 3) and not short_only:
            prompt = (prompt * ((1024 + len(prompt) - 1) // len(prompt)))[:1024]
        specs.append(shared.Arrival(f"r{i}", name, prompt, caps[i % 4], budgets[i % 4],
            arrival_s=0 if i < concurrency else (i - concurrency + 1) * arrival_ms / 1000,
            arrival_step=0 if i < concurrency else i - concurrency + 1))
    return specs


def configure(engine, args, concurrency):
    if not engine.is_finished():
        raise RuntimeError("configuration changes require an idle engine")
    engine.configure_jetspec(args.draft, max_admissions_per_step=2)
    engine._jetspec_scheduler.max_num_seqs = concurrency
    runtime = engine._jetspec_scheduler.runtime
    flags = {"lightweight": args.lightweight, "batched_draft": args.batched_draft,
             "feature_storage": args.feature_storage}
    if hasattr(runtime, "configure_optimizations"):
        runtime.configure_optimizations(**flags)
    elif any(flags.values()):
        raise ValueError("requested optimizations are absent from the imported baseline")
    return flags


class ResidenceTracker(AbstractContextManager):
    """Observe residency start/end in both versions, including recompute prefill."""
    def __init__(self, runtime):
        self.runtime, self.start, self.intervals, self.open, self.patches = runtime, 0.0, {}, {}, []

    def begin(self, request_id, now):
        if request_id not in self.open:
            self.open[request_id] = now

    def end(self, request_id, now):
        if request_id in self.open:
            self.intervals.setdefault(request_id, []).append((self.open.pop(request_id), now))

    def __enter__(self):
        original = self.runtime.create_request
        def create(*a, **k):
            before = time.perf_counter()
            result = original(*a, **k)
            self.begin(result.request_id, before)
            return result
        self.patch("create_request", create)
        for name in ("suspend", "finish", "cancel"):
            if hasattr(self.runtime, name):
                method = getattr(self.runtime, name)
                def terminal(request, *a, _method=method, **k):
                    result = _method(request, *a, **k)
                    self.end(request.request_id, time.perf_counter())
                    return result
                self.patch(name, terminal)
        # resume may delegate to create_request. begin() is idempotent.
        if hasattr(self.runtime, "resume"):
            original_resume = self.runtime.resume
            def resume(*a, **k):
                before = time.perf_counter()
                if a and isinstance(a[0],dict) and "request_id" in a[0]:
                    self.begin(a[0]["request_id"],before)
                result = original_resume(*a, **k)
                self.begin(result.request_id, before)
                return result
            self.patch("resume", resume)
        return self

    def patch(self, name, function):
        self.patches.append((name, getattr(self.runtime, name)))
        setattr(self.runtime, name, function)

    def __exit__(self, *exc):
        for name, function in reversed(self.patches):
            setattr(self.runtime, name, function)


def serve(engine, specs, *, label, deadline=600, clock="wall", hook=None):
    import torch
    from nanovllm import SamplingParams
    runtime = engine._jetspec_scheduler.runtime
    records = {s.request_id: {"request_id": s.request_id, "prompt_id": s.prompt_id,
        "prompt_length": len(s.prompt), "max_tokens": s.max_tokens, "tree_budget": s.tree_budget,
        "scheduled_arrival_s": s.arrival_s, "token_ids": [], "first_token_s": None,
        "last_token_s": None, "terminal_s": None, "status": "not_arrived", "delivery_batches": []} for s in specs}
    ids, submitted, arrivals, steps, acceptance = {}, set(), [], [], []
    allocator = shared.AllocationObserver(engine)
    residence = ResidenceTracker(runtime)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    residence.start = start
    tick = 0
    next_progress = 30
    with allocator, residence:
        while True:
            elapsed = time.perf_counter() - start
            if elapsed > deadline:
                raise TimeoutError(f"{label} exceeded {deadline}s: last steps={steps[-2:]}")
            if elapsed >= next_progress:
                print(f"progress {label}: {elapsed:.1f}s steps={tick} emitted={sum(len(r['token_ids']) for r in records.values())}", flush=True)
                next_progress += 30
            for spec in specs:
                ready = elapsed >= spec.arrival_s if clock == "wall" else tick >= spec.arrival_step
                if spec.request_id in submitted or not ready:
                    continue
                active = not engine.is_finished()
                request_id = engine.add_request(spec.prompt,
                    SamplingParams(temperature=0, max_tokens=spec.max_tokens, ignore_eos=spec.ignore_eos),
                    request_id=f"{label}:{spec.request_id}", tree_budget=spec.tree_budget)
                ids[request_id] = spec.request_id
                submitted.add(spec.request_id)
                now = time.perf_counter() - start
                records[spec.request_id].update(submitted_s=now, engine_request_id=request_id, status="waiting")
                arrivals.append({"request_id": spec.request_id, "at_s": now, "at_step": tick,
                    "service_started": bool(steps), "engine_active_before_arrival": active})
            if engine.is_finished():
                if len(submitted) == len(specs):
                    break
                if clock == "wall":
                    time.sleep(0.005)
                tick += 1
                continue
            if hook:
                hook.before_step(records)
            before = time.perf_counter()
            engine.step()
            delivered = time.perf_counter() - start
            info = engine.last_step_info
            events = []
            for event in info["events"]:
                row = records[ids[event["request_id"]]]
                tokens = [int(v) for v in event["token_ids"]]
                kind = event["kind"]
                events.append({"request_id": event["request_id"], "kind": kind, "token_ids": tokens})
                if kind == "tokens":
                    row["token_ids"].extend(tokens)
                    if tokens:
                        row["first_token_s"] = delivered if row["first_token_s"] is None else row["first_token_s"]
                        row["last_token_s"] = delivered
                        row["delivery_batches"].append({"at_s": delivered, "tokens": len(tokens)})
                elif kind in ("finished", "cancelled", "error"):
                    if row["terminal_s"] is not None:
                        raise AssertionError("duplicate terminal event")
                    if row["token_ids"] != tokens:
                        raise AssertionError("terminal output differs from streamed delta history")
                    row.update(status=kind, terminal_s=delivered, metrics=shared.json_safe(event.get("metrics", {})))
                    if kind == "error":
                        raise RuntimeError(event.get("reason", "request terminal error"))
            verify = info.get("verification")
            verify_records = []
            if verify:
                for record in verify.get("requests", []):
                    entry = {"request_id": record["request_id"], "accepted_draft_length": record["accepted_draft_length"],
                        "committed_path_length": record.get("committed_path_length", len(record.get("committed_path_indices", []))),
                        "emitted_output_length": record.get("emitted_output_length", len(record.get("output_block", [])))}
                    verify_records.append(entry)
                    acceptance.append(entry)
            steps.append({"step": tick, "returned_s": delivered, "wall_s": time.perf_counter()-before,
                "events": events, "admitted_ids": list(info.get("admitted_ids", [])),
                "resumed_ids": list(info.get("resumed_ids", [])), "preempted_ids": list(info.get("preempted_ids", [])),
                "capacity": shared.json_safe(info.get("capacity", {})), "blocked": info.get("blocked", False),
                "node_counts": list((verify or {}).get("node_counts", [])),
                "capacity_during_verify": shared.json_safe((verify or {}).get("capacity_during_verify", {})),
                "verification_requests": verify_records})
            if hook:
                hook.after_step(records)
            tick += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    ordered = [records[s.request_id] for s in specs]
    for row in ordered:
        if row["terminal_s"] is None or len(row["token_ids"]) != row["max_tokens"]:
            raise AssertionError("missing terminal or controlled output cap not reached")
        n = len(row["token_ids"])
        row["submitted_ttft_s"] = row["first_token_s"] - row["submitted_s"] if n else None
        row["offered_ttft_s"] = row["first_token_s"] - row["scheduled_arrival_s"] if n and clock=="wall" else None
        row["submitted_e2e_s"] = row["terminal_s"] - row["submitted_s"]
        row["offered_e2e_s"] = row["terminal_s"] - row["scheduled_arrival_s"] if clock=="wall" else None
        row["arrival_submission_lag_s"] = row["submitted_s"] - row["scheduled_arrival_s"] if clock=="wall" else None
        row["delivery_tpot_s"] = (row["last_token_s"]-row["first_token_s"])/(n-1) if n>1 else None
        row["submitted_amortized_output_s"] = row["submitted_e2e_s"]/n if n else None
        row["offered_amortized_output_s"] = row["offered_e2e_s"]/n if n and clock=="wall" else None
        intervals = residence.intervals.get(row["engine_request_id"], [])
        row["residence_intervals_s"] = [(a-start,b-start) for a,b in intervals]
        row["service_residence_s"] = sum(b-a for a,b in intervals)
        row["nonresident_wait_s"] = max(0,row["submitted_e2e_s"]-row["service_residence_s"])
    metric_names = ["submitted_ttft_s", "offered_ttft_s", "submitted_e2e_s", "offered_e2e_s",
        "arrival_submission_lag_s", "delivery_tpot_s", "submitted_amortized_output_s",
        "offered_amortized_output_s", "service_residence_s", "nonresident_wait_s"]
    return {"wall_s": wall, "actual_output_tokens": sum(len(r["token_ids"]) for r in ordered),
        "tokens_per_second": sum(len(r["token_ids"]) for r in ordered)/wall,
        "metrics": {name: distribution([r[name] for r in ordered if r[name] is not None]) for name in metric_names},
        "requests": ordered, "arrivals": arrivals, "steps": steps,
        "dynamic_live_arrival_seen": any(a["service_started"] and a["engine_active_before_arrival"] for a in arrivals),
        "peak_allocator_used_blocks": allocator.peak_blocks,
        "peak_reserved_kv_slots": allocator.peak_blocks*runtime.block_size,
        "peak_live_kv_slots": max((s["capacity"].get("live_kv_slots",0) for s in steps),default=0),
        "max_packed_requests": max((len(s["node_counts"]) for s in steps),default=0),
        "accepted_draft_length": distribution([r["accepted_draft_length"] for r in acceptance]),
        "effective_output_per_verification": distribution([r["emitted_output_length"] for r in acceptance]),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "capacity_after": engine._jetspec_scheduler.capacity_snapshot()}


class StageProbe(shared.ServingProbe):
    def __init__(self, engine):
        super().__init__(engine)
        self.feature_updates = []
        self.draft_batches = []

    def __enter__(self):
        super().__enter__()
        for name, stage in (("_propose_drafts","draft_batch_stage"), ("_build_trees","trees_batch_stage"),
                            ("_accept_batch","accept_batch_stage")):
            if hasattr(self.runner, name):
                original = getattr(self.runner,name)
                def wrapped(*a,_original=original,_stage=stage,**k):
                    result = self.measure(_stage,_original,*a,**k)
                    if _stage=="draft_batch_stage":
                        proposer = getattr(self.runner,"_batch_proposer",None)
                        if proposer is not None:
                            self.draft_batches.append(shared.json_safe(dict(proposer.last_stats)))
                    return result
                self.patch(self.runner,name,wrapped)
        from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata
        if hasattr(PackedTreeMetadata,"from_host_trees"):
            original_host_metadata = PackedTreeMetadata.from_host_trees
            self.patch(PackedTreeMetadata,"from_host_trees",classmethod(
                lambda cls,*a,**k:self.measure("host_metadata_build",original_host_metadata,*a,**k)))
        if hasattr(self.runner,"resume"):
            original_resume = self.runner.resume
            self.patch(self.runner,"resume",lambda *a,**k:self.measure("recompute_prefill",original_resume,*a,**k))
        if hasattr(self.state_module.PagedTargetState,"_publish_features"):
            original_publish = self.state_module.PagedTargetState._publish_features
            self.patch(self.state_module.PagedTargetState,"_publish_features",lambda state,*a,**k:
                self.measure("feature_publish",original_publish,state,*a,**k))
        cls = self.state_module.BatchTreeTransaction
        original = cls._prepare_commit
        def prepare(transaction, state, nodes, hidden, path, maximum, tokens, **kwargs):
            before = state.target_hidden
            result = self.measure("feature_and_metadata_prepare", original,transaction,state,nodes,hidden,path,maximum,tokens,**kwargs)
            after = result.next_hidden
            shared_storage = before.untyped_storage().data_ptr()==after.untyped_storage().data_ptr()
            accepted_bytes = int(path.numel())*int(hidden.shape[-1])*hidden.element_size()
            feature_plan = getattr(result,"features",getattr(result,"feature_plan",None))
            append_bytes = int(feature_plan.append_bytes) if feature_plan is not None else accepted_bytes
            history_bytes = int(feature_plan.history_bytes) if feature_plan is not None else int(before.numel()*before.element_size())
            self.feature_updates.append({"prefix_tokens":int(before.shape[1]),"accepted_tokens":int(path.numel()),
                "logical_prefix_bytes":before.numel()*before.element_size(),"accepted_payload_bytes":accepted_bytes,
                "logical_output_bytes":after.numel()*after.element_size(),"same_backing":shared_storage,
                "estimated_materialized_feature_bytes":accepted_bytes if shared_storage else after.numel()*after.element_size(),
                "feature_append_copy_bytes":append_bytes,"feature_history_copy_bytes":history_bytes,
                "feature_update_copy_bytes":append_bytes+history_bytes,
                "copy_counter_source":"feature append plan" if feature_plan is not None else "baseline torch.cat tensor payload"})
            return result
        self.patch(cls,"_prepare_commit",prepare)
        return self

    def metrics(self):
        result = super().metrics()
        result["feature_updates"] = self.feature_updates
        result["estimated_feature_bytes_materialized"] = sum(r["estimated_materialized_feature_bytes"] for r in self.feature_updates)
        result["accepted_feature_payload_bytes"] = sum(r["accepted_payload_bytes"] for r in self.feature_updates)
        result["feature_append_copy_bytes"] = sum(r["feature_append_copy_bytes"] for r in self.feature_updates)
        result["feature_history_copy_bytes"] = sum(r["feature_history_copy_bytes"] for r in self.feature_updates)
        result["feature_update_copy_bytes"] = sum(r["feature_update_copy_bytes"] for r in self.feature_updates)
        proposer = getattr(self.runner,"_batch_proposer",None)
        result["batched_draft_stats"] = shared.json_safe(getattr(proposer,"last_stats",None))
        result["draft_batches"] = self.draft_batches
        result["draft_batched_forward_calls"] = sum(r.get("batched_forward_calls",0) for r in self.draft_batches)
        result["draft_serial_forward_calls"] = sum(r.get("serial_forward_calls",0) for r in self.draft_batches)
        result["feature_traffic_note"] = "tensor payload estimate, not a hardware memory bandwidth counter; read+write traffic is larger"
        return result


class KernelWindow:
    def __init__(self, output, threshold=128, steps=4):
        import torch
        self.torch,self.output,self.threshold,self.steps = torch,output,threshold,steps
        self.profile,self.active,self.done,self.count = None,False,False,0

    def before_step(self, records):
        if not self.done and not self.active and max((len(r["token_ids"]) for r in records.values()),default=0)>=self.threshold:
            self.profile = self.torch.profiler.profile(activities=[self.torch.profiler.ProfilerActivity.CPU,
                                                                  self.torch.profiler.ProfilerActivity.CUDA])
            self.profile.start()
            self.active = True

    def after_step(self, records):
        if self.active:
            self.count += 1
            if self.count>=self.steps:
                self.profile.stop()
                self.active,self.done = False,True

    def metrics(self):
        if self.active:
            self.profile.stop()
            self.active,self.done = False,True
        if self.profile is None:
            return {"captured":False,"reason":"output threshold not reached"}
        trace = str(Path(self.output).with_suffix(".trace.json"))
        self.profile.export_chrome_trace(trace)
        cpu = [{"operator":e.key,"calls":e.count,"self_cpu_ms":e.self_cpu_time_total/1000,
                "total_cpu_ms":e.cpu_time_total/1000} for e in self.profile.key_averages()]
        kernels = {}
        for event in self.profile.events():
            if event.device_type==self.torch.autograd.DeviceType.CUDA:
                row = kernels.setdefault(event.name,{"kernel":event.name,"calls":0,"device_ms":0.0})
                row["calls"] += 1
                row["device_ms"] += event.time_range.elapsed_us()/1000
        return {"captured":True,"steps":self.count,"threshold_outputs":self.threshold,"trace_path":trace,
            "top_cpu_ops":sorted(cpu,key=lambda r:r["self_cpu_ms"],reverse=True)[:30],
            "top_cuda_kernels":sorted(kernels.values(),key=lambda r:r["device_ms"],reverse=True)[:30],
            "cpu_sync_self_ms":sum(r["self_cpu_ms"] for r in cpu if "Synchronize" in r["operator"]),
            "cpu_launch_self_ms":sum(r["self_cpu_ms"] for r in cpu if "LaunchKernel" in r["operator"]),
            "all_observed_cuda_device_ms":sum(r["device_ms"] for r in kernels.values()),
            "device_time_note":"CUDA activity includes kernels and any observed device copy/memset events"}


def pressure(engine,args,prompts):
    configure(engine,args,2)
    runtime,manager = engine._jetspec_scheduler.runtime,engine.scheduler.block_manager
    runtime.release_idle_scratch()
    specs = [shared.Arrival(f"pressure-{i}",name,(prompts[name]["prompt_token_ids"]*16)[:250],8,63)
             for i,name in enumerate(("natural_language","math_logic"))]
    held = manager.reserve_provisional(max(0,len(manager.free_block_ids)-3))
    try:
        result = serve(engine,specs,label=f"{args.variant}-pressure",deadline=args.deadline,clock="step")
        result["preemption_seen"] = any(s["preempted_ids"] for s in result["steps"])
        result["resume_seen"] = any(s["resumed_ids"] for s in result["steps"])
        result["foreign_pages_preserved"] = set(held).issubset(manager.used_block_ids)
        result["usable_pages"] = 3
        return result
    finally:
        manager.release_provisional(held)
        runtime.release_idle_scratch()


def summarize(samples):
    result = {}
    for case in dict.fromkeys(s["case"] for s in samples):
        rows = [s for s in samples if s["case"]==case]
        names = rows[0]["metrics"]
        result[case] = {"samples":len(rows),"tok_s_median":statistics.median(r["tokens_per_second"] for r in rows),
            "metrics":{name:distribution([request[name] for row in rows for request in row["requests"]
                if request[name] is not None]) for name in names},
            "max_packed_requests":max(r["max_packed_requests"] for r in rows),
            "peak_reserved_kv_slots":max(r["peak_reserved_kv_slots"] for r in rows),
            "all_live_arrival":all(r["dynamic_live_arrival_seen"] for r in rows)}
    return result


def compare(report, baseline_path):
    """Refuse nonmatched results instead of silently presenting a speedup."""
    baseline = json.loads(Path(baseline_path).read_text())
    if report.get("worktree_status","").strip():
        raise ValueError("formal candidate comparison requires a clean optimized checkout")
    if (baseline.get("kind")!="phase4_matched_jetspec" or
            not baseline.get("revision","").startswith(BASELINE) or
            baseline.get("worktree_status","").strip() or
            any(baseline.get("optimization_flags",{}).values())):
        raise ValueError("comparison baseline must be clean exact 7bcb754 JetSpec with all optimization flags false")
    for name in ("source_fingerprint_unchanged","allocator_clean_after_disable","all_timed_requests_finished"):
        if baseline.get(name) is not True or report.get(name) is not True:
            raise ValueError(f"cannot compare an incomplete or failed run: {name}")
    if baseline.get("script_sha256")!=report.get("script_sha256") or baseline.get("benchmark_helper_sha256")!=report.get("benchmark_helper_sha256"):
        raise ValueError("comparison requires the identical frozen harness and helpers")
    config_keys = ("tensor_parallel_size","enforce_eager","gpu_memory_utilization","max_num_batched_tokens",
        "max_model_len","max_num_seqs","kvcache_block_size","arrival_ms","expected_pool_blocks","warmup")
    if any(baseline["config"].get(k)!=report["config"].get(k) for k in config_keys):
        raise ValueError("comparison hardware/serving configuration differs")
    if baseline.get("pool")!=report.get("pool") or baseline.get("environment")!=report.get("environment"):
        raise ValueError("comparison fixed pool, GPU, torch or CUDA environment differs")
    if baseline.get("models") is None or baseline.get("models")!=report.get("models"):
        raise ValueError("comparison Target/Draft paths, configs, indexes or checkpoint shard manifest differs")
    official = lambda r:{k:v for k,v in r["production_file_sha256"].items() if k.startswith("official_jetspec:")}
    if not official(baseline) or official(baseline)!=official(report):
        raise ValueError("comparison official JetSpec/Draft source differs")
    cases = {}
    for case, candidate_summary in report.get("summary",{}).items():
        old_rows = [row for row in baseline["samples"] if row["case"]==case]
        new_rows = [row for row in report["samples"] if row["case"]==case]
        if not old_rows:
            raise ValueError(f"baseline has no matching workload case: {case}")
        hashes = {row["workload_sha256"] for row in old_rows+new_rows}
        totals = {row["actual_output_tokens"] for row in old_rows+new_rows}
        if len(hashes)!=1 or len(totals)!=1:
            raise ValueError(f"offered workload or actual controlled output count differs: {case}")
        if not all(row["dynamic_live_arrival_seen"] for row in old_rows+new_rows):
            raise ValueError(f"no genuine in-service arrival was observed: {case}")
        baseline_summary = baseline["summary"][case]
        numeric = []
        # Different wall-clock delivery schedules and changed Draft/GEMM shapes
        # are documented, not conflated with a same-shape correctness oracle.
        for i,new in enumerate(new_rows):
            old = old_rows[min(i,len(old_rows)-1)]
            old_requests = {r["request_id"]:r for r in old["requests"]}
            for request in new["requests"]:
                reference = old_requests[request["request_id"]]
                numeric.append({"repeat":new["repeat"],"request_id":request["request_id"],
                    "exact":request["token_ids"]==reference["token_ids"],
                    "first_divergence":shared.first_difference(reference["token_ids"],request["token_ids"])})
        cases[case] = {"baseline_samples":len(old_rows),"candidate_samples":len(new_rows),
            "workload_sha256":next(iter(hashes)),"actual_output_tokens":next(iter(totals)),
            "baseline_tok_s":baseline_summary["tok_s_median"],"candidate_tok_s":candidate_summary["tok_s_median"],
            "speed_ratio":candidate_summary["tok_s_median"]/baseline_summary["tok_s_median"],
            "baseline_metrics":baseline_summary["metrics"],"candidate_metrics":candidate_summary["metrics"],
            "baseline_peak_reserved_kv_slots":baseline_summary["peak_reserved_kv_slots"],
            "candidate_peak_reserved_kv_slots":candidate_summary["peak_reserved_kv_slots"],
            "different_schedule_numeric_comparisons":numeric}
    return {"baseline_path":str(Path(baseline_path).resolve()),
        "baseline_revision":baseline["revision"],"baseline_production_source_sha256":baseline["production_source_sha256"],
        "candidate_production_source_sha256":report["production_source_sha256"],
        "harness_sha256":report["script_sha256"],"strict_workload_config_pool_checks_passed":True,"cases":cases}


def run(args):
    sys.path.insert(0,str(Path(args.repo).resolve()))
    import torch,nanovllm
    if not Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()):
        raise RuntimeError("nano import escaped the requested checkout")
    source = identity(args)
    if args.require_revision and (not source["revision"].startswith(args.require_revision) or source["worktree_status"].strip()):
        raise RuntimeError("baseline must be a clean exact required revision")
    if args.compare and source["worktree_status"].strip():
        raise RuntimeError("formal comparison refuses a dirty candidate checkout")
    torch.manual_seed(0)
    prompts = {p["id"]:p for p in json.loads(Path(args.oracle).read_text())["prompts"]}
    config = {"tensor_parallel_size":1,"enforce_eager":True,"gpu_memory_utilization":0.8,
              "max_num_batched_tokens":4096,"max_model_len":4096,"max_num_seqs":16,"kvcache_block_size":256}
    engine = nanovllm.LLM(args.target,**config)
    configure(engine,args,8)
    pool = engine.model_runner.kv_cache
    if int(pool.shape[2])!=args.expected_pool_blocks:
        raise AssertionError(f"fixed pool mismatch: observed {pool.shape[2]}, required {args.expected_pool_blocks}")
    report = {"schema_version":1,"kind":"phase4_matched_jetspec","variant":args.variant,**source,
        "models":{"target":model_identity(args.target),"draft":model_identity(args.draft)},
        "optimization_flags":{"lightweight":args.lightweight,"batched_draft":args.batched_draft,"feature_storage":args.feature_storage},
        "config":{**config,"warmup":args.warmup,"repeats":args.repeats,"deadline_s":args.deadline,
                  "arrival_ms":args.arrival_ms,"expected_pool_blocks":args.expected_pool_blocks},
        "environment":{"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name()},
        "pool":{"blocks":int(pool.shape[2]),"slots":int(pool.shape[2]*pool.shape[3]),"bytes":pool.numel()*pool.element_size(),
                "shape":list(pool.shape),"dtype":str(pool.dtype)},
        "notes":["matched baseline is Phase3.2 JetSpec vs Phase4 JetSpec, not an ordinary-AR baseline",
            "same wall-clock offered arrivals; actual shape/schedule may differ and cross-shape bitwise AR equality is not required",
            "delivery TPOT=(last-first delivered token time)/(N-1); burst tokens share a step-return timestamp",
            "offered/submitted amortized time per output includes TTFT and is reported separately from TPOT",
            "service residence starts at create/resume begin, includes prefill, and excludes suspended host waiting",
            "timed samples include common host event/arrival/residency bookkeeping and O(1) allocator observation",
            "stage CUDA elapsed includes host gaps; nested stages are not additive; kernel window is a separate diagnostic"],
        "samples":[],"profiles":[],"pressure":None}
    save_json(args.output,report)
    try:
        for concurrency in ([] if args.skip_timing else [int(v) for v in args.concurrencies.split(",")]):
            for output in [int(v) for v in args.outputs.split(",")]:
                specs = workload(prompts,concurrency,output,args.arrival_ms,short_only=args.short_only)
                case = f"{'short' if args.short_only else 'mixed'}-c{concurrency}-o{output}"
                workload_hash = hashlib.sha256(json.dumps([asdict(s) for s in specs],sort_keys=True).encode()).hexdigest()
                for warmup in range(args.warmup):
                    configure(engine,args,concurrency)
                    serve(engine,specs,label=f"{args.variant}-{case}-w{warmup}",deadline=args.deadline)
                for repeat in range(args.repeats):
                    configure(engine,args,concurrency)
                    sample = serve(engine,specs,label=f"{args.variant}-{case}-r{repeat}",deadline=args.deadline)
                    sample.update(case=case,concurrency=concurrency,output_cap=output,repeat=repeat,workload_sha256=workload_hash)
                    report["samples"].append(sample)
                    report["summary"] = summarize(report["samples"])
                    save_json(args.output,report)
                    print(f"sample {args.variant} {case} r{repeat}: {sample['tokens_per_second']:.3f} tok/s actual={sample['actual_output_tokens']} wall={sample['wall_s']:.3f}s",flush=True)
        if args.profile:
            configure(engine,args,8)
            specs = workload(prompts,8,args.profile_output,args.arrival_ms)
            with StageProbe(engine) as probe:
                profiled = serve(engine,specs,label=f"{args.variant}-stage",deadline=args.deadline)
                stages = probe.metrics()
            save_json(args.output,{**report,"profiles":[{"kind":"stage","run":profiled,"profile":stages}]})
            report["profiles"].append({"kind":"stage","run":profiled,"profile":stages})
            configure(engine,args,8)
            window = KernelWindow(args.output,args.kernel_threshold,args.kernel_steps)
            kernel_run = serve(engine,specs,label=f"{args.variant}-kernels",deadline=args.deadline,hook=window)
            report["profiles"].append({"kind":"kernel_window","run":kernel_run,"profile":window.metrics()})
            save_json(args.output,report)
        if args.pressure:
            report["pressure"] = pressure(engine,args,prompts)
        report["all_timed_requests_finished"] = all(r["status"]=="finished" for s in report["samples"] for r in s["requests"])
    finally:
        scheduler = getattr(engine,"_jetspec_scheduler",None)
        if scheduler is not None:
            for request_id in list(scheduler.requests):
                engine.cancel_request(request_id)
            scheduler.drain_events()
        engine.disable_jetspec()
        report["allocator_clean_after_disable"] = not engine.scheduler.block_manager.used_block_ids
        report["fingerprint_after"] = identity(args)
        report["source_fingerprint_unchanged"] = all(source[k]==report["fingerprint_after"][k]
            for k in ("production_source_sha256","script_sha256","benchmark_helper_sha256"))
        save_json(args.output,report)
    assert report["allocator_clean_after_disable"] and report["source_fingerprint_unchanged"]
    if args.compare:
        report["comparison"] = compare(report,args.compare)
        save_json(args.output,report)
    print(json.dumps({"variant":args.variant,"summary":report.get("summary",{}),
        "allocator_clean":report["allocator_clean_after_disable"],"source_unchanged":report["source_fingerprint_unchanged"]},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--variant",default="baseline")
    parser.add_argument("--require-revision",default="")
    parser.add_argument("--target",default=shared.TARGET)
    parser.add_argument("--draft",default=shared.DRAFT)
    parser.add_argument("--oracle",default=shared.ORACLE)
    parser.add_argument("--concurrencies",default="1,2,4,8")
    parser.add_argument("--outputs",default="128,512")
    parser.add_argument("--warmup",type=int,default=1)
    parser.add_argument("--repeats",type=int,default=1)
    parser.add_argument("--deadline",type=float,default=600)
    parser.add_argument("--arrival-ms",type=float,default=20)
    parser.add_argument("--expected-pool-blocks",type=int,default=249)
    parser.add_argument("--short-only",action="store_true")
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--profile-output",type=int,default=512)
    parser.add_argument("--kernel-threshold",type=int,default=128)
    parser.add_argument("--kernel-steps",type=int,default=4)
    parser.add_argument("--pressure",action="store_true")
    parser.add_argument("--skip-timing",action="store_true")
    parser.add_argument("--compare",help="strictly matched clean-7bcb754 baseline artifact")
    parser.add_argument("--lightweight",action="store_true")
    parser.add_argument("--feature-storage",action="store_true")
    parser.add_argument("--batched-draft",action="store_true")
    args = parser.parse_args()
    if any(int(v)<1 for v in args.outputs.split(",")) or any(not 1<=int(v)<=16 for v in args.concurrencies.split(",")):
        parser.error("outputs must be positive and concurrency in [1,16]")
    if args.warmup<0 or args.repeats<1 or args.deadline<=180 or args.kernel_steps<1:
        parser.error("warmup>=0, repeats>=1, deadline>180 and kernel steps>=1 required")
    run(args)


if __name__=="__main__":
    main()
