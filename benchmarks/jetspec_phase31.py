#!/usr/bin/env python3
"""Lightweight Phase-3 sanity and Phase-3.1 packed-ragged qualification.

The sanity worker imports nano exclusively from the requested clean, pinned
repository. Production code is never patched; diagnostic wrappers only observe
allocation, commit, and scheduler boundaries. Timings are not a formal serving
benchmark and include eager Python orchestration and request cleanup.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from jetspec_phase3 import Probe, distribution, save_json


TARGET = "/root/autodl-tmp/models/modelscope-Qwen3-8B"
DRAFT = "/root/autodl-tmp/models/jetspec-qwen3-8b-020a198caefde24a2891ad827cba7fb977ccdc36"
ORACLE = "/root/autodl-tmp/benchmarks/jetspec-phase0/oracle.json"
PHASE3_REVISION = "58433e73d2aa14f112c4a4d1048ec4ff790063f0"


def first_divergence(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {"index": index, "left_token": a, "right_token": b}
    if len(left) != len(right):
        return {"index": min(len(left), len(right)), "left_length": len(left), "right_length": len(right)}
    return None


def identity(args):
    revision = subprocess.check_output(["git", "-C", args.repo, "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", args.repo, "status", "--porcelain"], text=True)
    diff = subprocess.check_output(["git", "-C", args.repo, "diff", "HEAD", "--binary"])
    repository = Path(args.repo).resolve()
    production_paths = subprocess.check_output(["rg", "--files", "-g", "*.py", str(repository / "nanovllm")], text=True).splitlines()
    hashes = {str(Path(path).relative_to(repository)): hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for path in production_paths}
    helper = Path(__file__).resolve().parent / "jetspec_phase3.py"
    hashes["loaded_benchmark_helper:jetspec_phase3.py"] = hashlib.sha256(helper.read_bytes()).hexdigest()
    spec = importlib.util.find_spec("jetspec")
    if spec is not None and spec.origin:
        official = Path(spec.origin).resolve().parent
        official_paths = subprocess.check_output(["rg", "--files", "-g", "*.py", str(official)], text=True).splitlines()
        hashes.update({"official_jetspec:" + str(Path(path).relative_to(official)):
                       hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in official_paths})
    return {"repository": args.repo, "revision": revision, "worktree_status": status,
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
            "production_file_sha256": hashes,
            "production_source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def sanity(args):
    source = identity(args)
    if source["revision"] != PHASE3_REVISION or source["worktree_status"].strip():
        raise RuntimeError("sanity requires a clean exact Phase-3.0 detached checkout")
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.speculative.jetspec.state import PagedTargetState

    torch.manual_seed(0)
    prompts = json.loads(Path(args.oracle).read_text())["prompts"]
    config = {"tensor_parallel_size": 1, "enforce_eager": True,
              "gpu_memory_utilization": 0.8, "max_num_batched_tokens": 4096,
              "max_model_len": 4096, "max_num_seqs": 1, "kvcache_block_size": 256}
    engine = LLM(args.target, **config)
    manager = engine.scheduler.block_manager
    probe = Probe(torch, manager, PagedTargetState)
    ordinary_boundaries = []
    tracing_ordinary = False
    original_postprocess = engine.scheduler.postprocess

    def postprocess(seqs, token_ids, is_prefill):
        if tracing_ordinary:
            live = sum(seq.num_cached_tokens + seq.num_scheduled_tokens for seq in seqs)
            reserved = len(manager.used_block_ids) * manager.block_size
            ordinary_boundaries.append({"live_slots": live,
                "used_blocks": len(manager.used_block_ids), "leased_capacity_slots": reserved,
                "amplification": reserved / max(live, 1), "is_prefill": is_prefill})
        return original_postprocess(seqs, token_ids, is_prefill)

    engine.scheduler.postprocess = postprocess

    def generate(mode, prompt_ids, count):
        if mode == "ordinary":
            return engine.generate([prompt_ids], SamplingParams(temperature=0, max_tokens=count,
                                                               ignore_eos=False), use_tqdm=False)[0]
        return engine.generate_jetspec(prompt_ids, args.draft, max_tokens=count,
                                       tree_backend="paged", return_rounds=True)

    # Resident Target and Draft, identical global KV pool, then one full matched
    # warmup pass for each mode. Initialization is excluded from sample timing.
    generate("jetspec", prompts[0]["prompt_token_ids"], 32)
    runtime = engine._jetspec_runtime[1]
    probe.install_forward(runtime)
    for _ in range(args.warmup):
        for prompt in prompts:
            for mode in ("ordinary", "jetspec"):
                generate(mode, prompt["prompt_token_ids"], 32)
    torch.cuda.synchronize()
    pool = engine.model_runner.kv_cache
    report = {"schema_version": 1, "kind": "lightweight_phase30_sanity", **source,
              "config": {**config, "dtype": "bfloat16", "temperature": 0,
                         "ignore_eos": False, "warmup_passes": args.warmup,
                         "repeats": args.repeats, "tree_depth": 15, "tree_width": 7, "tree_budget": 63},
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "target": args.target, "draft": args.draft},
              "global_pool": {"blocks": pool.shape[2], "slots": pool.shape[2] * 256,
                              "bytes": pool.numel() * pool.element_size()},
              "timing_scope": "CUDA-synchronized full generate call including cleanup; model loading/warmups excluded",
              "notes": ["ordinary and JetSpec use different numerical and attention backends",
                        "Draft remains resident during both measurements", "not a formal performance phase",
                        "global preallocated pool bytes are distinct from leased-page capacity",
                        "allocator/commit observation wrappers add small Python/event overhead"],
              "samples": [], "comparisons": [], "long_attempts": []}

    def sample(mode, prompt_id, prompt_ids, count, repeat, *, long_run=False):
        nonlocal tracing_ordinary, ordinary_boundaries
        probe.reset()
        ordinary_boundaries = []
        tracing_ordinary = mode == "ordinary"
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        generated = generate(mode, prompt_ids, count)
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        tracing_ordinary = False
        metrics = probe.metrics()
        if mode == "ordinary":
            metrics["round_boundaries"] = ordinary_boundaries
        record = {"mode": mode, "prompt_id": prompt_id, "prompt_tokens": len(prompt_ids),
                  "requested_tokens": count, "generated_tokens": len(generated["token_ids"]),
                  "token_ids": generated["token_ids"], "repeat": repeat, "long_run": long_run,
                  "wall_latency_s": wall, "tok_s": len(generated["token_ids"]) / wall,
                  "eos_before_limit": len(generated["token_ids"]) < count,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "allocator_used_after": len(manager.used_block_ids), "probe": metrics,
                  "text_preview": generated["text"][:500]}
        if mode == "jetspec":
            record["runtime"] = {key: value for key, value in generated.items()
                                 if key not in ("token_ids", "text", "qualification_probes")}
        if manager.used_block_ids or (mode == "jetspec" and not generated["state_invariant_passed"]):
            raise AssertionError("request did not cleanly restore allocator/state")
        report["samples"].append(record)
        save_json(args.output, report)
        print(f"{mode} {prompt_id} limit={count} actual={record['generated_tokens']} tok/s={record['tok_s']:.2f}", flush=True)
        return record

    for repeat in range(args.repeats):
        for prompt in prompts:
            ordinary = sample("ordinary", prompt["id"], prompt["prompt_token_ids"], 32, repeat)
            jetspec = sample("jetspec", prompt["id"], prompt["prompt_token_ids"], 32, repeat)
            expected = prompt["target"]["token_ids"][:32]
            report["comparisons"].append({"prompt_id": prompt["id"], "repeat": repeat,
                "ordinary_oracle_exact": ordinary["token_ids"] == expected,
                "jetspec_oracle_exact": jetspec["token_ids"] == expected,
                "ordinary_jetspec_first_divergence": first_divergence(ordinary["token_ids"], jetspec["token_ids"]),
                "jetspec_vs_ordinary_speed_ratio": jetspec["tok_s"] / ordinary["tok_s"]})

    candidates = [
        "Continue the following list of consecutive positive integers, one per line, up to 1000. "
        "Do not summarize, omit numbers, or stop early.\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n",
        "Write a numbered list of 200 concrete tips for reproducible machine-learning research. "
        "Each item needs two complete sentences. Continue the full list without summary.\n1.",
        "Write a detailed technical tutorial on transformer inference and KV cache management. "
        "Cover 20 sections, each with several long paragraphs, examples, and practical trade-offs. "
        "Do not conclude early; continue until the generation limit.\nSection 1: Introduction\n",
    ]
    long_succeeded = False
    if not args.skip_long:
        for candidate_id, candidate in enumerate(candidates):
            ids = engine.tokenizer.encode(candidate)
            for mode in ("ordinary", "jetspec"):
                generate(mode, ids, 32)
            pair = []
            for count in (256, 512):
                ordinary = sample("ordinary", f"long_natural_{candidate_id}", ids, count, 0, long_run=True)
                jetspec = sample("jetspec", f"long_natural_{candidate_id}", ids, count, 0, long_run=True)
                pair.append({"requested_tokens": count, "ordinary_actual": ordinary["generated_tokens"],
                             "jetspec_actual": jetspec["generated_tokens"],
                             "first_divergence": first_divergence(ordinary["token_ids"], jetspec["token_ids"]),
                             "jetspec_vs_ordinary_speed_ratio": jetspec["tok_s"] / ordinary["tok_s"]})
            long_succeeded = all(p["ordinary_actual"] == p["requested_tokens"]
                                 and p["jetspec_actual"] == p["requested_tokens"] for p in pair)
            report["long_attempts"].append({"candidate_id": candidate_id, "prompt": candidate,
                                            "results": pair, "real_256_and_512_reached": long_succeeded})
            save_json(args.output, report)
            if long_succeeded:
                break

    fixed = [s for s in report["samples"] if not s["long_run"]]
    report["summary"] = {"jetspec_oracle_passed": all(c["jetspec_oracle_exact"] for c in report["comparisons"]),
        "ordinary_oracle_passed": all(c["ordinary_oracle_exact"] for c in report["comparisons"]),
        "real_long_256_and_512_reached": long_succeeded if not args.skip_long else None,
        "allocator_clean": not manager.used_block_ids,
        "fixed_ordinary_tok_s_median": statistics.median(s["tok_s"] for s in fixed if s["mode"] == "ordinary"),
        "fixed_jetspec_tok_s_median": statistics.median(s["tok_s"] for s in fixed if s["mode"] == "jetspec"),
        "fixed_speed_ratio_median": statistics.median(c["jetspec_vs_ordinary_speed_ratio"] for c in report["comparisons"])}
    save_json(args.output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)


class BatchProbe:
    """Observe the packed round and one all-layer accepted-only copy."""

    def __init__(self, torch, runner, manager, state_module):
        self.torch, self.runner, self.manager = torch, runner, manager
        self.diagnostic = False
        self.capture_margins = False
        self.frozen_all_rounds = True
        self.last_normalized_hidden = None

        def capture_head_input(module, arguments, output):
            if self.capture_margins:
                self.last_normalized_hidden = arguments[0].detach()

        runner.target.lm_head.register_forward_hook(capture_head_input)
        self.reset()
        original_allocate = manager._allocate_block

        def allocate():
            block = original_allocate()
            self.peak_blocks = max(self.peak_blocks, len(manager.used_block_ids))
            self.allocations += 1
            return block

        manager._allocate_block = allocate
        original_verify = runner._verify_batch

        def verify(requests, trees, transaction, metadata):
            if self.diagnostic:
                arena = transaction.arena
                committed = {b for state in transaction.states for b in state.owned_blocks}
                if committed.intersection(arena.blocks):
                    raise AssertionError("packed scratch and committed pages overlap")
                if sum(nodes.numel() for nodes in transaction.node_slots) != transaction.packed_node_slots.numel():
                    raise AssertionError("packed tree partition does not cover the query batch")
                arena.kv_pool[:, :, arena.blocks] = float("nan")
                self.scratch_signatures.append(tuple(arena.blocks))
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = original_verify(requests, trees, transaction, metadata)
            end.record()
            self.verify_events.append((start, end))
            self.verify_shapes.append([int(nodes.numel()) for nodes in transaction.node_slots])
            if self.capture_margins:
                self.record_margins(requests, trees, metadata, result[0])
            if self.diagnostic:
                logits, taps = result[:2]
                if not bool(torch.isfinite(logits).all().item()) or not bool(torch.isfinite(taps).all().item()):
                    raise AssertionError("NaN padding/stale scratch leaked into packed outputs")
                if self.frozen_all_rounds or not self.frozen_done:
                    self.frozen_done = True
                    self.frozen_isolation(original_verify, requests, trees, transaction, metadata, result)
            return result

        runner._verify_batch = verify
        original_copy = state_module.copy_accepted_kv

        def copy(pool, source, destination, block_size):
            before = None
            if self.diagnostic:
                if bool(torch.isin(source, destination).any().item()):
                    raise AssertionError("accepted source and canonical destination overlap")
                before = self.read(pool, source, block_size).clone()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            payload = original_copy(pool, source, destination, block_size)
            end.record()
            self.copy_events.append((start, end))
            self.copy_payload_bytes += payload
            self.copy_counts.append(int(source.numel()))
            if self.diagnostic:
                after = self.read(pool, destination, block_size)
                exact = torch.equal(before.contiguous().view(torch.uint8), after.contiguous().view(torch.uint8))
                self.raw_copy_checks.append({"all_layer_k_and_v_bytes_exact": exact,
                                            "accepted_slots": source.numel(), "payload_bytes": payload})
                if not exact:
                    raise AssertionError("packed accepted K/V bytes changed on physical commit")
            return payload

        state_module.copy_accepted_kv = copy
        transaction_class = state_module.BatchTreeTransaction
        original_commit = transaction_class.commit

        def commit(transaction, node_hidden, accepted_paths, committed_tokens):
            previous = []
            if self.diagnostic:
                for state in transaction.states:
                    previous.append((state.logical_slots.clone(), self.read(state.kv_pool, state.logical_slots,
                                                                             state.block_size).clone()))
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = original_commit(transaction, node_hidden, accepted_paths, committed_tokens)
            end.record()
            self.commit_events.append((start, end))
            self.boundaries.append(transaction.capacity_snapshot())
            if self.diagnostic:
                for state, nodes, path, (old_slots, history) in zip(transaction.states, transaction.node_slots,
                                                                  accepted_paths, previous):
                    selected = torch.zeros(nodes.numel(), dtype=torch.bool, device=nodes.device)
                    selected[path.long()] = True
                    rejected_excluded = not bool(torch.isin(nodes[~selected], state.logical_slots).any().item())
                    history_exact = torch.equal(history.contiguous().view(torch.uint8),
                        self.read(state.kv_pool, old_slots, state.block_size).contiguous().view(torch.uint8))
                    canonical_unchanged = torch.equal(old_slots, state.logical_slots[:old_slots.numel()])
                    state.assert_round_invariant()
                    check = {"rejected_excluded": rejected_excluded, "historical_kv_bytes_unchanged": history_exact,
                             "historical_mapping_unchanged": canonical_unchanged,
                             "accepted_root_inclusive": path.numel()}
                    self.request_checks.append(check)
                    if not all((rejected_excluded, history_exact, canonical_unchanged)):
                        raise AssertionError(f"packed request isolation failed: {check}")
            return result

        transaction_class.commit = commit

    @staticmethod
    def read(pool, slots, block_size):
        return pool[:, :, slots // block_size, slots % block_size]

    def frozen_isolation(self, original_verify, requests, trees, transaction, metadata, base):
        """Keep total Q, row positions and masks fixed while perturbing inputs."""
        torch = self.torch
        from jetspec.tree import gpu_tree_accept

        base_logits, base_taps = base[:2]
        for chosen, request in enumerate(requests):
            selected_rows = metadata.request_slice(chosen)
            if len(requests) > 1:
                changed = [copy.copy(tree) for tree in trees]
                for other, tree in enumerate(changed):
                    if other != chosen:
                        tree.token_ids = torch.zeros_like(tree.token_ids)
                logits, taps = original_verify(requests, changed, transaction, metadata)[:2]
                exact_logits = torch.equal(base_logits[selected_rows], logits[selected_rows])
                exact_taps = torch.equal(base_taps[selected_rows], taps[selected_rows])
                check = {"kind": "other_requests_all_tree_tokens_changed", "request_id": request.request_id,
                         "total_query_tokens": metadata.total_queries, "node_counts": list(metadata.node_counts_host),
                         "request_logits_bitwise_exact": exact_logits, "request_taps_bitwise_exact": exact_taps}
                self.frozen_checks.append(check)
                if not exact_logits or not exact_taps:
                    raise AssertionError(f"frozen full-Q inter-request isolation failed: {check}")

            # Off-path rows remain present in the GEMM; change their input tokens
            # and isolate their attention rows without changing the accepted rows.
            tree = trees[chosen]
            path, _, _ = gpu_tree_accept(tree.token_ids, base_logits[selected_rows].argmax(-1),
                                        tree.parent_indices, tree.depth, max_depth=self.runner.tree_depth)
            off_path = torch.ones(tree.num_nodes, dtype=torch.bool, device=tree.token_ids.device)
            off_path[path.long()] = False
            changed = [copy.copy(candidate) for candidate in trees]
            changed[chosen].token_ids = tree.token_ids.clone()
            changed[chosen].token_ids[off_path] = 0
            changed_bias = metadata.qq_bias.clone()
            begin = int(metadata.qq_bias_offsets[chosen].item())
            n = int(tree.num_nodes)
            matrix = changed_bias[begin:begin + n * n].view(n, n)
            matrix[off_path] = float("-inf")
            off_rows = off_path.nonzero(as_tuple=False).flatten()
            matrix[off_rows, off_rows] = 0.0
            changed_meta = replace(metadata, qq_bias=changed_bias)
            logits, taps = original_verify(requests, changed, transaction, changed_meta)[:2]
            path_rows = path.long() + metadata.query_offsets[chosen]
            exact_logits = torch.equal(base_logits.index_select(0, path_rows), logits.index_select(0, path_rows))
            exact_taps = torch.equal(base_taps.index_select(0, path_rows), taps.index_select(0, path_rows))
            check = {"kind": "off_path_tokens_changed_and_rows_self_only", "request_id": request.request_id,
                     "total_query_tokens": metadata.total_queries, "node_counts": list(metadata.node_counts_host),
                     "accepted_path": path.tolist(), "path_logits_bitwise_exact": exact_logits,
                     "path_taps_bitwise_exact": exact_taps}
            self.frozen_checks.append(check)
            if not exact_logits or not exact_taps:
                raise AssertionError(f"frozen same-Q branch isolation failed: {check}")
        # Restore every source K/V before the production acceptance/copy resumes.
        restored = original_verify(requests, trees, transaction, metadata)
        if not torch.equal(base_logits, restored[0]) or not torch.equal(base_taps, restored[1]):
            raise AssertionError("restoring the original frozen batch changed its outputs")

    def record_margins(self, requests, trees, metadata, logits):
        torch = self.torch
        from jetspec.tree import gpu_tree_accept

        for request_index, (request, tree) in enumerate(zip(requests, trees)):
            rows = metadata.request_slice(request_index)
            if self.runner._reference_mode:
                path = torch.zeros(1, dtype=torch.long, device=logits.device)
            else:
                path, _, _ = gpu_tree_accept(tree.token_ids, logits[rows].argmax(-1),
                    tree.parent_indices, tree.depth, max_depth=self.runner.tree_depth)
            selected = logits[rows].index_select(0, path.long()).float()
            values, tokens = selected.topk(2, dim=-1)
            argmaxes = selected.argmax(-1).tolist()
            for node, top_values, top_tokens, argmax in zip(path.tolist(), values.tolist(), tokens.tolist(), argmaxes):
                output_index = len(request.output_ids) + int(tree.depth[node].item())
                if output_index >= request.max_new_tokens:
                    continue
                global_row = metadata.query_offsets[request_index] + node
                normalized_hidden = self.last_normalized_hidden[global_row].double()
                token_indices = torch.tensor(top_tokens, dtype=torch.long, device=logits.device)
                # Rescore only two candidate weight rows against the already-
                # computed BF16 normalized hidden. This is a diagnostic of head
                # rounding/upstream differences, not an alternative sampler.
                candidate_weights = self.runner.target.lm_head.weight.index_select(0, token_indices).double()
                candidate_logits = torch.mv(candidate_weights, normalized_hidden).tolist()
                self.node_margins.append({"request_id": request.request_id,
                    "prediction_output_index": output_index, "node_index": node,
                    "prefix_length": request.state.cache_len, "total_query_tokens": metadata.total_queries,
                    "node_counts": list(metadata.node_counts_host), "top2_token_ids": top_tokens,
                    "top2_logits": top_values, "top2_margin": top_values[0] - top_values[1],
                    "argmax_token_id": argmax, "topk_tie_order_need_not_equal_argmax_order": top_values[0] == top_values[1],
                    "top2_fp64_dots_from_existing_bf16_hidden_weights": candidate_logits,
                    "fp64_signed_candidate_gap": candidate_logits[0] - candidate_logits[1],
                    "fp64_preferred_of_only_two_candidates": top_tokens[0 if candidate_logits[0] >= candidate_logits[1] else 1]})

    def reset(self):
        self.peak_blocks = len(self.manager.used_block_ids)
        self.allocations = self.copy_payload_bytes = 0
        self.verify_events, self.commit_events, self.copy_events = [], [], []
        self.verify_shapes, self.boundaries, self.copy_counts = [], [], []
        self.raw_copy_checks, self.request_checks, self.scratch_signatures = [], [], []
        self.frozen_checks, self.frozen_done = [], False
        self.node_margins = []

    def metrics(self):
        gpu_times = lambda events: [s.elapsed_time(e) / 1000 for s, e in events]
        return {"peak_used_blocks": self.peak_blocks,
                "peak_reserved_slots": self.peak_blocks * self.manager.block_size,
                "allocation_calls": self.allocations, "copy_payload_bytes": self.copy_payload_bytes,
                "accepted_copy_slots_by_round": list(self.copy_counts),
                "verify_gpu_by_round_s": gpu_times(self.verify_events),
                "commit_gpu_by_round_s": gpu_times(self.commit_events),
                "copy_gpu_by_round_s": gpu_times(self.copy_events),
                "tree_sizes_by_round": copy.deepcopy(self.verify_shapes), "capacity_curve": copy.deepcopy(self.boundaries),
                "raw_copy_checks": copy.deepcopy(self.raw_copy_checks), "request_checks": copy.deepcopy(self.request_checks),
                "scratch_signatures": list(self.scratch_signatures), "frozen_full_query_checks": copy.deepcopy(self.frozen_checks),
                "node_argmax_margins": copy.deepcopy(self.node_margins)}


def batch(args):
    source = identity(args)
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.speculative.jetspec.batch_runtime import JetSpecBatchRuntime
    from nanovllm.speculative.jetspec import state as state_module

    torch.manual_seed(0)
    prompts = json.loads(Path(args.oracle).read_text())["prompts"]
    by_id = {p["id"]: p for p in prompts}
    config = {"tensor_parallel_size": 1, "enforce_eager": True,
              "gpu_memory_utilization": 0.8, "max_num_batched_tokens": 4096,
              "max_model_len": 4096, "max_num_seqs": 4, "kvcache_block_size": 256}
    engine = LLM(args.target, **config)
    manager = engine.scheduler.block_manager
    runner = JetSpecBatchRuntime(target=engine.model_runner.model, tokenizer=engine.tokenizer,
        draft_model=args.draft, kv_pool=engine.model_runner.kv_cache, block_manager=manager,
        block_size=256, max_model_len=4096)
    probe = BatchProbe(torch, runner, manager, state_module)
    cases = [{"name": f"c1_{p['id']}", "prompt_ids": [p["id"]], "budgets": [63]} for p in prompts]
    cases += [
        {"name": "c2_equal", "prompt_ids": ["natural_language", "math_logic"], "budgets": [63, 63]},
        {"name": "c2_ragged", "prompt_ids": ["natural_language", "math_logic"], "budgets": [63, 31]},
        {"name": "c3_ragged_63_31_47", "prompt_ids": ["natural_language", "math_logic", "long_prompt"],
         "budgets": [63, 31, 47]},
        {"name": "c4_equal", "prompt_ids": ["natural_language", "math_logic", "long_prompt", "long_continuation"],
         "budgets": [63, 63, 63, 63]},
        {"name": "c4_ragged", "prompt_ids": ["natural_language", "math_logic", "long_prompt", "long_continuation"],
         "budgets": [63, 31, 47, 63]},
    ]
    if args.cases:
        selected = set(args.cases.split(","))
        cases = [case for case in cases if case["name"] in selected]
    if not cases:
        raise ValueError("no selected batch cases")
    pool = engine.model_runner.kv_cache
    report = {"schema_version": 1, "kind": "phase31_packed_ragged_qualification", **source,
              "config": {**config, "dtype": "bfloat16", "temperature": 0, "max_new_tokens": 32,
                         "warmup_passes": args.warmup, "repeats": args.repeats},
              "global_pool": {"blocks": pool.shape[2], "slots": pool.shape[2] * 256,
                              "bytes": pool.numel() * pool.element_size()},
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "target": args.target, "draft": args.draft},
              "notes": ["same packed backend/per-request-budget AR comparison is separate from timed samples; Q schedules may differ",
                        "full-layer byte checks and full scratch NaN poison are untimed diagnostics",
                        "runner arena remains leased between requests and is counted in reserved capacity",
                        "ordinary samples retain the idle runner arena; request-only peak excludes that lease",
                        "performance is eager single-process prototype, not continuous serving",
                        "FP64 diagnostic rescoring reads only two head weight rows; production head/argmax never changed"],
              "samples": [], "correctness": [], "diagnostics": [], "exception_cleanup": None,
              "engine_api_smoke": None}

    def inputs(case):
        return [by_id[pid]["prompt_token_ids"] for pid in case["prompt_ids"]]

    def generate(case):
        return runner.generate_batch(inputs(case), max_new_tokens=32, tree_budgets=case["budgets"],
                                     ignore_eos=False, return_rounds=True)

    try:
        for _ in range(args.warmup):
            for case in cases:
                generate(case)
                engine.generate(inputs(case), SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)
        for repeat in range(args.repeats):
            for case in cases:
                for mode in ("ordinary", "packed_jetspec"):
                    probe.reset()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if mode == "ordinary":
                        output = engine.generate(inputs(case), SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)
                        generated = {"requests": output}
                    else:
                        generated = generate(case)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - start
                    total = sum(len(req["token_ids"]) for req in generated["requests"])
                    sample = {"case": case["name"], "mode": mode, "repeat": repeat,
                        "concurrency": len(case["budgets"]), "tree_budgets": case["budgets"],
                        "generated_tokens": total, "wall_latency_s": wall, "aggregate_tok_s": total / wall,
                        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "allocator_used_after": len(manager.used_block_ids),
                        "idle_runner_scratch_blocks": len(runner.arena.blocks) if mode == "ordinary" else 0,
                        "request_peak_used_blocks_excluding_idle_runner_arena":
                            probe.peak_blocks - (len(runner.arena.blocks) if mode == "ordinary" else 0),
                        "probe": probe.metrics(), "result": generated}
                    report["samples"].append(sample)
                    save_json(args.output, report)
                    print(f"{mode} {case['name']} actual={total} tok/s={total / wall:.2f}", flush=True)

        for case in cases:
            probe.reset()
            probe.capture_margins = True
            packed = generate(case)
            packed_margins = copy.deepcopy(probe.node_margins)
            if not hasattr(runner, "generate_target_batch"):
                raise RuntimeError("batch runtime must provide the same-packed-backend/per-request-budget AR reference")
            probe.reset()
            reference = runner.generate_target_batch(inputs(case), max_new_tokens=32,
                                                     tree_budgets=case["budgets"], ignore_eos=False)
            reference_margins = copy.deepcopy(probe.node_margins)
            probe.capture_margins = False
            checks = []
            for prompt_id, actual, expected in zip(case["prompt_ids"], packed["requests"], reference["requests"]):
                oracle = by_id[prompt_id]["target"]["token_ids"][:32]
                divergence = first_divergence(actual["token_ids"], expected["token_ids"])
                position = divergence["index"] if divergence else None
                packed_margin = next((m for m in packed_margins if m["request_id"] == actual["request_id"]
                                      and m["prediction_output_index"] == position), None)
                reference_margin = next((m for m in reference_margins if m["request_id"] == expected["request_id"]
                                         and m["prediction_output_index"] == position), None)
                checks.append({"prompt_id": prompt_id,
                    "packed_shape_ar_exact": actual["token_ids"] == expected["token_ids"],
                    "packed_shape_ar_first_divergence": divergence,
                    "first_divergence_packed_margin": packed_margin,
                    "first_divergence_reference_margin": reference_margin,
                    "original_c1_oracle_exact": actual["token_ids"] == oracle,
                    "original_c1_oracle_first_divergence": first_divergence(actual["token_ids"], oracle),
                    "packed_token_ids": actual["token_ids"], "reference_token_ids": expected["token_ids"]})
            shape_schedule = lambda result: [{"node_counts": r["node_counts"],
                "total_query_tokens": r["total_query_tokens"], "cu_seqlens_q": r["cu_seqlens_q"]} for r in result["rounds"]]
            report["correctness"].append({"case": case["name"], "checks": checks,
                "packed_shape_schedule": shape_schedule(packed), "reference_shape_schedule": shape_schedule(reference),
                "packed_argmax_margins": packed_margins, "reference_argmax_margins": reference_margins,
                "reference_shape_note": "per-live-request rows match budget; removal schedules and tree/chain reduction groups can differ"})
            probe.reset()
            probe.diagnostic = True
            poisoned = generate(case)
            torch.cuda.synchronize()
            probe.diagnostic = False
            metrics = probe.metrics()
            poison_exact = [r["token_ids"] for r in poisoned["requests"]] == [r["token_ids"] for r in packed["requests"]]
            if not poison_exact:
                raise AssertionError("poison/reused scratch changed packed generation tokens")
            report["diagnostics"].append({"case": case["name"], "poison_generation_exact": poison_exact,
                "raw_all_layer_copy_exact": all(c["all_layer_k_and_v_bytes_exact"] for c in metrics["raw_copy_checks"]),
                "request_history_and_rejected_isolation": all(all((c["rejected_excluded"],
                    c["historical_kv_bytes_unchanged"], c["historical_mapping_unchanged"])) for c in metrics["request_checks"]),
                "frozen_full_query_isolation_passed": all(all(v for k, v in check.items() if k.endswith("bitwise_exact"))
                                                          for check in metrics["frozen_full_query_checks"]),
                "probe": metrics})
            save_json(args.output, report)

        # Route through the actual LLM entry point using the already loaded head.
        engine._jetspec_batch_runtime = ((args.draft, 15, 7, 63), runner)
        smoke_prompts = [by_id[key]["prompt_token_ids"] for key in ("natural_language", "math_logic", "long_prompt")]
        smoke_first = engine.generate_jetspec_batch(smoke_prompts, args.draft, max_tokens=4,
                                                    tree_budgets=[63, 31, 47])
        cached_identity = engine.get_jetspec_batch_runtime(args.draft) is runner
        engine.generate([smoke_prompts[0]], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
        idle_returned = not manager.used_block_ids and not runner.arena.blocks
        smoke_second = engine.generate_jetspec_batch(smoke_prompts, args.draft, max_tokens=4,
                                                     tree_budgets=[63, 31, 47])
        rebuilt = bool(runner.arena.blocks)
        api_tokens_exact = [r["token_ids"] for r in smoke_first["requests"]] == [r["token_ids"] for r in smoke_second["requests"]]
        live = runner.create_request(smoke_prompts[0], max_new_tokens=4, tree_budget=63)
        ordinary_rejected = False
        try:
            engine.generate([smoke_prompts[0]], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
        except RuntimeError:
            ordinary_rejected = True
        finally:
            runner.cancel(live)
        report["engine_api_smoke"] = {"cached_runner_identity_same": cached_identity,
            "ordinary_generation_returned_idle_arena_to_allocator": idle_returned,
            "subsequent_packed_call_rebuilt_arena": rebuilt, "packed_tokens_exact_after_arena_rebuild": api_tokens_exact,
            "ordinary_generation_rejected_while_packed_request_live": ordinary_rejected}
        if not all(report["engine_api_smoke"].values()):
            raise AssertionError(f"real LLM API lifecycle smoke failed: {report['engine_api_smoke']}")

        # The failure is injected after real verification kernels wrote the arena.
        original_verify = runner._verify_batch

        def fail_after_verify(*positional, **keywords):
            original_verify(*positional, **keywords)
            raise RuntimeError("phase31 injected exception after packed GPU verification")

        runner._verify_batch = fail_after_verify
        try:
            generate(cases[-1])
            report["exception_cleanup"] = {"injected_exception_seen": False}
        except RuntimeError as exc:
            torch.cuda.synchronize()
            report["exception_cleanup"] = {"injected_exception_seen": "phase31 injected" in str(exc),
                "error": str(exc), "used_blocks_after_error": len(manager.used_block_ids),
                "live_requests_cleared": not runner.requests,
                "active_transaction_cleared": runner._active_transaction is None,
                "arena_lease_inactive": not runner.arena.active,
                "only_runner_arena_pages_remain": manager.used_block_ids == set(runner.arena.blocks)}
        finally:
            runner._verify_batch = original_verify
        recovery = generate(cases[0])
        report["exception_cleanup"]["subsequent_generation_succeeded"] = bool(recovery["requests"])
        first_sample = next(s for s in report["samples"] if s["case"] == cases[0]["name"] and s["mode"] == "packed_jetspec")
        report["exception_cleanup"]["subsequent_generation_tokens_exact"] = (
            [r["token_ids"] for r in recovery["requests"]] == [r["token_ids"] for r in first_sample["result"]["requests"]])
    finally:
        runner.close()
        torch.cuda.synchronize()
        report["allocator_clean_after_close"] = not manager.used_block_ids
        save_json(args.output, report)
    report["summary"] = {
        "c1_oracle_requests_checked": sum(len(entry["checks"]) for entry in report["correctness"] if entry["case"].startswith("c1_")),
        "c1_original_oracle_exact": all(check["original_c1_oracle_exact"] for entry in report["correctness"]
                                        if entry["case"].startswith("c1_") for check in entry["checks"]),
        "packed_shape_ar_all_exact": all(check["packed_shape_ar_exact"] for entry in report["correctness"] for check in entry["checks"]),
        "packed_shape_ar_requests_checked": sum(len(entry["checks"]) for entry in report["correctness"]),
        "packed_shape_ar_exact_requests": sum(check["packed_shape_ar_exact"] for entry in report["correctness"] for check in entry["checks"]),
        "poison_and_raw_diagnostics_passed": all(entry["poison_generation_exact"] and entry["raw_all_layer_copy_exact"]
                                                and entry["request_history_and_rejected_isolation"]
                                                and entry["frozen_full_query_isolation_passed"] for entry in report["diagnostics"]),
        "allocator_clean_after_close": report["allocator_clean_after_close"],
        "real_engine_api_lifecycle_passed": all(report["engine_api_smoke"].values()),
        "exception_cleanup_and_recovery_passed": all(report["exception_cleanup"].get(key, False) for key in (
            "injected_exception_seen", "live_requests_cleared", "active_transaction_cleared",
            "arena_lease_inactive", "only_runner_arena_pages_remain", "subsequent_generation_tokens_exact")),
    }
    performance = {}
    for case in cases:
        modes = {}
        for mode in ("ordinary", "packed_jetspec"):
            selected = [s for s in report["samples"] if s["case"] == case["name"] and s["mode"] == mode]
            modes[mode] = {"samples": len(selected),
                "aggregate_tok_s_median": statistics.median(s["aggregate_tok_s"] for s in selected),
                "wall_latency_s": distribution([s["wall_latency_s"] for s in selected]),
                "peak_used_blocks": max(s["probe"]["peak_used_blocks"] for s in selected),
                "request_peak_blocks_excluding_idle_arena": max(s["request_peak_used_blocks_excluding_idle_runner_arena"] for s in selected),
                "verify_gpu_per_round_s": distribution([v for s in selected for v in s["probe"]["verify_gpu_by_round_s"]]),
                "commit_gpu_per_round_s": distribution([v for s in selected for v in s["probe"]["commit_gpu_by_round_s"]]),
                "copy_gpu_per_round_s": distribution([v for s in selected for v in s["probe"]["copy_gpu_by_round_s"]]),
                "copy_payload_bytes_per_sample": [s["probe"]["copy_payload_bytes"] for s in selected]}
        modes["packed_vs_ordinary_median_tok_s_ratio"] = (
            modes["packed_jetspec"]["aggregate_tok_s_median"] / modes["ordinary"]["aggregate_tok_s_median"])
        performance[case["name"]] = modes
    report["summary"]["performance_by_case"] = performance
    ending_source = identity(args)
    report["source_fingerprint_after"] = ending_source
    report["source_fingerprint_unchanged"] = (
        source["production_source_sha256"] == ending_source["production_source_sha256"]
        and source["script_sha256"] == ending_source["script_sha256"]
    )
    save_json(args.output, report)
    if not report["source_fingerprint_unchanged"]:
        raise AssertionError("production or benchmark source changed while the worker was running")
    print(json.dumps({key: value for key, value in report["summary"].items()
                      if key != "performance_by_case"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sanity", "batch"), default="sanity")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--draft", default=DRAFT)
    parser.add_argument("--oracle", default=ORACLE)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--cases", default="")
    args = parser.parse_args()
    if args.mode == "sanity":
        sanity(args)
    else:
        batch(args)


if __name__ == "__main__":
    main()
