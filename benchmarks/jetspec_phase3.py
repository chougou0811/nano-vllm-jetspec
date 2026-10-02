#!/usr/bin/env python3
"""Matched c1 Phase-3 benchmark and opt-in real-model KV diagnostics.

Run with the Phase-0 virtualenv. The two repositories are imported in isolated
subprocesses; the script itself may live in either repository. Diagnostics are
separate from timed samples. No production numerical path is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = "/root/autodl-tmp/models/modelscope-Qwen3-8B"
DEFAULT_DRAFT = "/root/autodl-tmp/models/jetspec-qwen3-8b-020a198caefde24a2891ad827cba7fb977ccdc36"
DEFAULT_ORACLE = "/root/autodl-tmp/benchmarks/jetspec-phase0/oracle.json"


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def distribution(values):
    return {"count": len(values), "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95), "min": min(values) if values else None,
            "max": max(values) if values else None}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class Probe:
    """Exact allocator counters and identical GPU seams in old and new code."""

    def __init__(self, torch, manager, state_class):
        self.torch, self.manager = torch, manager
        self.diagnostic = False
        self.reset()
        allocate, deallocate = manager._allocate_block, manager._deallocate_block

        def tracked_allocate():
            block = allocate()
            self.allocations += 1
            self.peak_blocks = max(self.peak_blocks, len(manager.used_block_ids) - self.before)
            return block

        def tracked_deallocate(block):
            result = deallocate(block)
            self.releases += 1
            return result

        manager._allocate_block, manager._deallocate_block = tracked_allocate, tracked_deallocate
        original_commit = state_class.commit_tree_path

        def commit(state, *args, **kwargs):
            nodes = args[0] if args else kwargs["node_slots"]
            path = args[2] if len(args) > 2 else kwargs["accepted_path"]
            source = None
            if self.diagnostic:
                source_slots = nodes.index_select(0, path.long())
                source = self.read(state, source_slots).clone()
                old_slots = state.logical_slots.clone()
                history = self.read(state, old_slots).clone()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            host_start = time.perf_counter()
            result = original_commit(state, *args, **kwargs)
            host_s = time.perf_counter() - host_start
            end.record()
            self.commit_events.append((start, end))
            self.commit_host_s.append(host_s)
            live = state.cache_len
            blocks = len(manager.used_block_ids) - self.before
            self.boundaries.append({"live_slots": live, "used_blocks": blocks,
                                    "leased_capacity_slots": blocks * state.block_size,
                                    "amplification": blocks * state.block_size / max(live, 1)})
            if self.diagnostic:
                destination = state.logical_slots[-path.numel():]
                exact = torch.equal(source.contiguous().view(torch.uint8),
                                    self.read(state, destination).contiguous().view(torch.uint8))
                rejected = torch.ones(nodes.numel(), dtype=torch.bool, device=nodes.device)
                rejected[path.long()] = False
                excluded = not bool(torch.isin(nodes[rejected], state.logical_slots).any().item())
                unchanged = torch.equal(old_slots, state.logical_slots[:old_slots.numel()])
                historical_bytes_unchanged = torch.equal(
                    history.contiguous().view(torch.uint8),
                    self.read(state, old_slots).contiguous().view(torch.uint8))
                self.raw_checks.append({"all_layer_k_and_v_bytes_exact": exact,
                                        "rejected_slots_excluded": excluded,
                                        "historical_slot_mapping_unchanged": unchanged,
                                        "historical_kv_bytes_unchanged": historical_bytes_unchanged,
                                        "path_length_root_inclusive": path.numel(),
                                        "kv_payload_bytes": source.numel() * source.element_size(),
                                        "physical_copy": not torch.equal(source_slots, destination)})
                if not (exact and excluded and unchanged and historical_bytes_unchanged):
                    raise AssertionError(f"raw KV commit diagnostic failed: {self.raw_checks[-1]}")
            return result

        state_class.commit_tree_path = commit

    def reset(self):
        self.before = len(self.manager.used_block_ids)
        self.peak_blocks = self.allocations = self.releases = 0
        self.boundaries, self.raw_checks = [], []
        self.commit_events, self.verify_events, self.commit_host_s = [], [], []
        self.scratch_signatures = []

    @staticmethod
    def read(state, slots):
        return state.kv_pool[:, :, slots // state.block_size, slots % state.block_size]

    def install_forward(self, runtime):
        original = runtime._target_forward_paged

        def forward(token_ids, positions, node_slots, logical_slots, qq_bias):
            if self.diagnostic:
                state = runtime._active_state
                block_size = runtime.block_size
                tree_blocks = (node_slots // block_size).unique()
                committed_blocks = (state.logical_slots // block_size).unique()
                if bool(self.torch.isin(tree_blocks, committed_blocks).any().item()):
                    raise AssertionError("scratch pages overlap committed pages")
                # Poison the full backing pages, including padding. Target verify
                # must overwrite every visible node, and not read padded/stale KV.
                state.kv_pool[:, :, tree_blocks] = float("nan")
                self.scratch_signatures.append(tuple(int(x) for x in tree_blocks.tolist()))
            start, end = (self.torch.cuda.Event(enable_timing=True),
                          self.torch.cuda.Event(enable_timing=True))
            start.record()
            result = original(token_ids, positions, node_slots, logical_slots, qq_bias)
            end.record()
            self.verify_events.append((start, end))
            return result

        runtime._target_forward_paged = forward

    def metrics(self):
        return {"peak_used_blocks": self.peak_blocks,
                "peak_leased_capacity_slots": self.peak_blocks * self.manager.block_size,
                "allocation_calls": self.allocations, "release_calls": self.releases,
                "round_boundaries": [dict(boundary) for boundary in self.boundaries],
                "verify_method_gpu_by_round_s": [a.elapsed_time(b) / 1000 for a, b in self.verify_events],
                "commit_method_gpu_by_round_s": [a.elapsed_time(b) / 1000 for a, b in self.commit_events],
                "commit_method_host_by_round_s": list(self.commit_host_s),
                "raw_kv_checks": [dict(check) for check in self.raw_checks],
                "scratch_block_signatures": list(self.scratch_signatures)}


def worker(args):
    # Do not import nano before selecting the repository: importing our checkout
    # while claiming to run the baseline would silently invalidate the comparison.
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    from nanovllm import LLM
    from nanovllm.speculative.jetspec.state import PagedTargetState

    torch.manual_seed(0)
    oracle = json.loads(Path(args.oracle).read_text())
    prompts = oracle["prompts"]
    if args.prompt_ids:
        selected = set(args.prompt_ids.split(","))
        prompts = [p for p in prompts if p["id"] in selected]
    if not prompts:
        raise ValueError("no selected oracle prompts")
    config = {"tensor_parallel_size": 1, "enforce_eager": True,
              "gpu_memory_utilization": args.gpu_memory_utilization,
              "max_num_batched_tokens": 4096, "max_model_len": 4096,
              "max_num_seqs": 1, "kvcache_block_size": 256}
    engine = LLM(args.target, **config)
    manager = engine.scheduler.block_manager
    probe = Probe(torch, manager, PagedTargetState)

    def generate(prompt):
        return engine.generate_jetspec(prompt["prompt_token_ids"], args.draft,
                                       max_tokens=args.max_tokens, tree_backend="paged",
                                       return_rounds=True)

    # Initialize the cached Draft runtime before measurement, then warm every
    # prompt. Both variants execute the same number of warmups in the same order.
    generate(prompts[0])
    runtime = engine._jetspec_runtime[1]
    probe.install_forward(runtime)
    for _ in range(args.warmup):
        for prompt in prompts:
            generate(prompt)
    torch.cuda.synchronize()
    kv_pool = engine.model_runner.kv_cache
    revision = subprocess.check_output(["git", "-C", args.repo, "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", args.repo, "status", "--porcelain"], text=True)
    diff = subprocess.check_output(["git", "-C", args.repo, "diff", "HEAD", "--binary"])
    result = {"schema_version": 1, "label": args.label, "repository": args.repo,
              "revision": revision, "worktree_dirty": bool(status.strip()),
              "worktree_status": status, "git_diff_head_sha256": hashlib.sha256(diff).hexdigest(),
              "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "config": {**config, "max_new_tokens": args.max_tokens,
              "tree_depth": 15, "tree_width": 7, "tree_budget": 63,
              "warmup_passes": args.warmup, "repeats": args.repeats,
              "temperature": 0.0, "seed": 0, "dtype": "bfloat16"},
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "python": sys.version,
              "target": args.target, "draft": args.draft, "oracle": args.oracle},
              "pool": {"blocks": kv_pool.shape[2], "slots": kv_pool.shape[2] * 256,
                       "bytes": kv_pool.numel() * kv_pool.element_size()},
              "timing_scope": {"e2e": "generate_jetspec call through cleanup, CUDA synchronized",
               "matched_verify": "_target_forward_paged entry to return, CUDA events",
               "matched_commit": "commit_tree_path entry to return, CUDA events",
               "runtime_times": "production runtime timing; may include different bookkeeping scopes",
               "diagnostics": "separate untimed generation with full-page NaN poison and raw KV checks"},
              "samples": [], "diagnostics": [], "padded_ar_comparator": [], "exception_cleanup": None}

    for repeat in range(args.repeats):
        for prompt in prompts:
            probe.reset()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = generate(prompt)
            torch.cuda.synchronize()
            wall = time.perf_counter() - start
            expected = prompt["target"]["token_ids"][:args.max_tokens]
            sample = {"repeat": repeat, "prompt_id": prompt["id"],
                      "prompt_tokens": len(prompt["prompt_token_ids"]),
                      "oracle_exact": generated["token_ids"] == expected,
                      "e2e_wall_s": wall, "actual_output_tokens": len(generated["token_ids"]),
                      "e2e_tokens_per_second": len(generated["token_ids"]) / wall,
                      "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "probe": probe.metrics(), "runtime": generated}
            result["samples"].append(sample)
            save_json(args.output, result)
            print(json.dumps({"label": args.label, "repeat": repeat, "prompt": prompt["id"],
                              "oracle_exact": sample["oracle_exact"], "e2e_s": wall}), flush=True)

    if args.diagnostics:
        probe.diagnostic = True
        for prompt in prompts:
            probe.reset()
            generated = generate(prompt)
            torch.cuda.synchronize()
            metrics = probe.metrics()
            signatures = metrics["scratch_block_signatures"]
            checks = {"oracle_exact": generated["token_ids"] == prompt["target"]["token_ids"][:args.max_tokens],
                      "all_layer_raw_kv_exact": all(c["all_layer_k_and_v_bytes_exact"] for c in metrics["raw_kv_checks"]),
                      "rejected_excluded": all(c["rejected_slots_excluded"] for c in metrics["raw_kv_checks"]),
                      "accepted_path_physically_copied": all(c["physical_copy"] for c in metrics["raw_kv_checks"]),
                      "historical_kv_unchanged": all(c["historical_kv_bytes_unchanged"] for c in metrics["raw_kv_checks"]),
                      "poisoned_full_scratch_pages_before_each_verify": True,
                      "scratch_backing_reused_within_request": len(signatures) > 1 and len(set(signatures)) < len(signatures),
                      "allocator_clean": not manager.used_block_ids,
                      "runtime_invariant": generated["state_invariant_passed"]}
            result["diagnostics"].append({"prompt_id": prompt["id"], "checks": checks,
                                         "probe": metrics, "token_ids": generated["token_ids"]})
            save_json(args.output, result)
        probe.diagnostic = False
        # Fail after verification has launched/written scratch, to exercise a real
        # GPU transaction abort rather than a CPU-only allocator failure.
        if hasattr(runtime, "abort") or "canonical" in result["samples"][0]["runtime"].get("lifecycle_design", ""):
            original = runtime._target_forward_paged

            def fail_after_verify(*positional, **keywords):
                original(*positional, **keywords)
                raise RuntimeError("phase3 injected exception after GPU verification")

            runtime._target_forward_paged = fail_after_verify
            try:
                generate(prompts[0])
                result["exception_cleanup"] = {"injected_exception_seen": False}
            except RuntimeError as exc:
                torch.cuda.synchronize()
                result["exception_cleanup"] = {"injected_exception_seen": "phase3 injected" in str(exc),
                                               "allocator_clean": not manager.used_block_ids,
                                               "active_state_cleared": runtime._active_state is None,
                                               "error": str(exc)}
            finally:
                runtime._target_forward_paged = original
            # A successful later request proves cleanup did not merely hide the
            # leaked lease or leave poisoned Draft/request state behind.
            recovery = generate(prompts[0])
            result["exception_cleanup"]["subsequent_generation_oracle_exact"] = recovery["token_ids"] == prompts[0]["target"]["token_ids"][:args.max_tokens]
            save_json(args.output, result)

    if args.padded_ar:
        for prompt in prompts:
            generated = runtime.generate_target_paged(prompt["prompt_token_ids"], args.max_tokens)
            result["padded_ar_comparator"].append({"prompt_id": prompt["id"],
                "oracle_exact": generated["token_ids"] == prompt["target"]["token_ids"][:args.max_tokens],
                "result": generated})

    verify_rounds, commit_rounds, host_rounds = [], [], []
    for sample in result["samples"]:
        verify_rounds.extend(sample["probe"]["verify_method_gpu_by_round_s"])
        commit_rounds.extend(sample["probe"]["commit_method_gpu_by_round_s"])
        host_rounds.extend(sample["probe"]["commit_method_host_by_round_s"])
    repeat_throughput = []
    for repeat in range(args.repeats):
        samples = [s for s in result["samples"] if s["repeat"] == repeat]
        repeat_throughput.append(sum(s["actual_output_tokens"] for s in samples) /
                                 sum(s["e2e_wall_s"] for s in samples))
    result["summary"] = {"oracle_exact_all_samples": all(s["oracle_exact"] for s in result["samples"]),
                         "aggregate_e2e_tokens_per_second": distribution(repeat_throughput),
                         "verify_round_gpu_s": distribution(verify_rounds),
                         "commit_round_gpu_s": distribution(commit_rounds),
                         "commit_round_host_s": distribution(host_rounds),
                         "peak_gpu_allocated_bytes": max(s["peak_gpu_allocated_bytes"] for s in result["samples"]),
                         "peak_gpu_reserved_bytes": max(s["peak_gpu_reserved_bytes"] for s in result["samples"]),
                         "peak_leased_capacity_slots": max(s["probe"]["peak_leased_capacity_slots"] for s in result["samples"])}
    passed = result["summary"]["oracle_exact_all_samples"]
    for diagnostic in result["diagnostics"]:
        if not all(diagnostic["checks"][key] for key in (
                "oracle_exact", "all_layer_raw_kv_exact", "rejected_excluded",
                "accepted_path_physically_copied", "historical_kv_unchanged",
                "allocator_clean", "runtime_invariant")):
            passed = False
        if len(diagnostic["probe"]["scratch_block_signatures"]) > 1 and not diagnostic["checks"]["scratch_backing_reused_within_request"]:
            passed = False
    if result["exception_cleanup"] is not None and not all(
            result["exception_cleanup"].get(key) for key in (
                "injected_exception_seen", "allocator_clean", "active_state_cleared",
                "subsequent_generation_oracle_exact")):
        passed = False
    if any(not entry["oracle_exact"] for entry in result["padded_ar_comparator"]):
        passed = False
    result["passed"] = passed
    save_json(args.output, result)
    print(json.dumps({"label": args.label, "passed": passed, "summary": result["summary"]}, indent=2), flush=True)
    if not passed:
        raise AssertionError("Phase-3 benchmark correctness gate failed; inspect JSON diagnostics")


def orchestrate(args):
    if not args.baseline_repo:
        raise ValueError("--baseline-repo must identify a detached checkout of 54d1636")
    baseline_revision = subprocess.check_output(["git", "-C", args.baseline_repo, "rev-parse", "HEAD"], text=True).strip()
    if baseline_revision != "54d1636cf4168e7ac7884d662d8e961d4b70ab2b":
        raise ValueError(f"baseline revision is not 54d1636: {baseline_revision}")
    output = Path(args.output).resolve()
    worker_results = {}
    for label, repo in (("baseline_54d1636", args.baseline_repo), ("phase3", args.repo)):
        destination = output.with_name(f"{output.stem}.{label}.json")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--repo", repo,
                   "--label", label, "--output", str(destination), "--target", args.target,
                   "--draft", args.draft, "--oracle", args.oracle, "--warmup", str(args.warmup),
                   "--repeats", str(args.repeats), "--max-tokens", str(args.max_tokens),
                   "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        if args.prompt_ids:
            command += ["--prompt-ids", args.prompt_ids]
        if args.diagnostics and label == "phase3":
            command.append("--diagnostics")
        if args.padded_ar:
            command.append("--padded-ar")
        env = dict(os.environ, PYTHONPATH=str(Path(repo).resolve()), TOKENIZERS_PARALLELISM="false")
        subprocess.run(command, env=env, check=True, cwd=repo)
        worker_results[label] = json.loads(destination.read_text())
    baseline, new = worker_results["baseline_54d1636"], worker_results["phase3"]
    comparison = {"schema_version": 1, "workers": worker_results,
                  "passed": baseline["passed"] and new["passed"],
                  "order": "baseline then phase3, isolated processes; same device/config/prompts/warmup/repeats",
                  "caveat": "fixed-order runs do not eliminate thermal/clock drift; repeat in reverse order for small deltas",
                  "per_prompt": []}
    for prompt_id in dict.fromkeys(s["prompt_id"] for s in baseline["samples"]):
        old_samples = [s for s in baseline["samples"] if s["prompt_id"] == prompt_id]
        new_samples = [s for s in new["samples"] if s["prompt_id"] == prompt_id]
        row = {"prompt_id": prompt_id}
        for label, samples in (("baseline", old_samples), ("phase3", new_samples)):
            gpu_commit = [sum(s["probe"]["commit_method_gpu_by_round_s"]) for s in samples]
            gpu_verify = [sum(s["probe"]["verify_method_gpu_by_round_s"]) for s in samples]
            row[label] = {"e2e_wall_s_median": statistics.median(s["e2e_wall_s"] for s in samples),
                          "verify_method_gpu_s_median": statistics.median(gpu_verify),
                          "commit_method_gpu_s_median": statistics.median(gpu_commit),
                          "peak_leased_capacity_slots": max(s["probe"]["peak_leased_capacity_slots"] for s in samples),
                          "round_end_capacity_slots": samples[0]["probe"]["round_boundaries"][-1]["leased_capacity_slots"],
                          "round_end_live_slots": samples[0]["probe"]["round_boundaries"][-1]["live_slots"],
                          "round_end_amplification": samples[0]["probe"]["round_boundaries"][-1]["amplification"],
                          "kv_copy_bytes": samples[0]["runtime"].get("kv_copy_bytes", 0),
                          "oracle_exact": all(s["oracle_exact"] for s in samples)}
        row["e2e_time_ratio_new_over_old"] = row["phase3"]["e2e_wall_s_median"] / row["baseline"]["e2e_wall_s_median"]
        row["matched_commit_gpu_extra_s"] = row["phase3"]["commit_method_gpu_s_median"] - row["baseline"]["commit_method_gpu_s_median"]
        comparison["per_prompt"].append(row)
    save_json(output, comparison)
    print(json.dumps(comparison["per_prompt"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--baseline-repo")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--oracle", default=DEFAULT_ORACLE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--label", default="phase3")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--prompt-ids", help="comma-separated subset of fixed oracle prompt IDs")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--padded-ar", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.max_tokens < 2 or args.max_tokens > 32:
        parser.error("require repeats>=1, warmup>=0, and 2<=max-tokens<=32 for fixed oracle")
    (worker if args.worker else orchestrate)(args)


if __name__ == "__main__":
    main()
