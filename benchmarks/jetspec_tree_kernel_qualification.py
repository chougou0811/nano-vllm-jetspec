#!/usr/bin/env python3
"""Untimed Phase-5 tree-attention gates; never a performance benchmark.

The FP32 accumulator is checked against independent parent-chain CPU FP64
attention at the frozen 1e-4 scaled-max and relative-RMS envelope. BF16
quantization is separately reported, with a same-input FP32-output round-trip
control. Full-model comparisons retain the Phase-4 2^-6 empirical BF16 gate.
No cross-shape autoregressive token-bitwise guarantee is introduced.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

import jetspec_phase4_qualification as frozen
import jetspec_phase32 as serving
import jetspec_chunked_prefill as chunks
from jetspec_numeric_oracles import FP32_ATTENTION_BOUND, argmax_flip_witness, fp64_attention, numerical_metrics, parent_chain
from jetspec_phase3 import save_json


PREFIX_LENGTHS = (0, 1, 63, 64, 65, 255, 256, 257, 1024, 2048)
BF16_BOUND = 2 ** -6
DEFAULT_CANDIDATE = "tree_attention:packed_tree_attention_gqa"


def candidate_entry(candidate=DEFAULT_CANDIDATE):
    """Explicit operator selection; never silently fall back to reference."""
    if ":" not in candidate:
        raise ValueError("candidate must be module:function")
    module_name, function_name = candidate.split(":", 1)
    if not module_name.startswith("nanovllm."):
        module_name = "nanovllm.speculative.jetspec." + module_name
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function) or "output_dtype" not in inspect.signature(function).parameters:
        raise ValueError("candidate must expose explicit output_dtype for pre-BF16 qualification")
    return module, function


def candidate_options(function, *, query_tile, num_warps, output_dtype=None):
    supported = inspect.signature(function).parameters
    options = {name: value for name, value in (("query_tile", query_tile), ("num_warps", num_warps))
               if name in supported}
    if output_dtype is not None:
        options["output_dtype"] = output_dtype
    return options


def launch_candidate(operands, candidate=DEFAULT_CANDIDATE, *, query_tile=1, num_warps=4, output_dtype=None):
    _module, function = candidate_entry(candidate)
    return function(*operands, **candidate_options(function, query_tile=query_tile,
                    num_warps=num_warps, output_dtype=output_dtype))


def candidate_identity(candidate, *, query_tile, num_warps):
    module, function = candidate_entry(candidate)
    path = Path(module.__file__).resolve()
    return {"entry": f"{module.__name__}:{function.__name__}", "module_path": str(path),
        "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "function_source_sha256": hashlib.sha256(inspect.getsource(function).encode()).hexdigest(),
        "supported_parameters": list(inspect.signature(function).parameters),
        "executed_options": candidate_options(function, query_tile=query_tile, num_warps=num_warps),
        "output_dtype_control": "same Q/K/V operand tensors and dtype; only output destination becomes FP32"}


def bits_equal(a, b):
    import torch
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.detach().contiguous().cpu().view(torch.uint8),
                        b.detach().contiguous().cpu().view(torch.uint8)))


def lifecycle_semantics(result, eos_ids):
    """Require actual EOS and cancel boundaries, not merely finite outputs."""
    rows = {row["request_id"]: row for row in result["requests"]}
    eos = rows["eos"]
    checks = {"real_eos_terminal": eos["status"] == "finished" and bool(eos["token_ids"]) and
              eos["token_ids"][-1] in eos_ids and len(eos["token_ids"]) < eos["max_tokens"],
              "running_cancel_terminal": rows["running-cancel"]["status"] == "cancelled",
              "queued_cancel_zero_output": rows["queued-cancel"]["status"] == "cancelled" and not rows["queued-cancel"]["token_ids"],
              "max_tokens_one_exact": rows["one-token"]["status"] == "finished" and len(rows["one-token"]["token_ids"]) == 1,
              "late_arrival_completed": rows["late"]["status"] == "finished" and rows["refill"]["status"] == "finished",
              "dynamic_live_arrival_seen": result["dynamic_live_arrival_seen"]}
    frozen.require(all(checks.values()), f"EOS/max-cap/cancel/dynamic serving gate failed: {checks}")
    return checks


def parents_for(nodes, seed=317):
    """Random topological tree; the oracle never reads production qq_bias."""
    import torch
    generator = torch.Generator().manual_seed(seed)
    return [-1] + [int(torch.randint(node, (1,), generator=generator)) for node in range(1, nodes)]


def mask_from_parents(parents):
    import torch
    mask = torch.zeros((len(parents), len(parents)), dtype=torch.bool)
    for node in range(len(parents)):
        mask[node, parent_chain(parents, node)] = True
    return mask


def synthetic_fixture(prefixes, counts=(63, 31, 47), *, device="cpu", dtype=None,
                      groups=4, head_dim=128, kv_heads=2, block_size=256, seed=317):
    """Disjoint, nonconsecutive prefix pages and randomly scattered tree slots.

    All valid operands are finite. Unleased pages and unused physical slots
    carry NaN deliberately; finite off-branch mutations are a separate gate.
    """
    import torch
    from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata
    if dtype is None:
        dtype = torch.bfloat16
    if len(prefixes) != len(counts):
        raise ValueError("prefix/count lengths differ")
    if groups < 1 or head_dim < 1 or kv_heads < 1:
        raise ValueError("invalid fixture head geometry")
    generator = torch.Generator().manual_seed(seed)
    prefix_pages = sum((prefix + block_size - 1) // block_size for prefix in prefixes)
    scratch_pages = (sum(counts) + block_size - 1) // block_size + 1
    total_pages = prefix_pages + scratch_pages + 3
    order = torch.randperm(total_pages, generator=generator).tolist()
    pages, prefix_slots, node_slots, parents, masks = [], [], [], [], []
    cursor = 0
    scratch_blocks = order[prefix_pages:prefix_pages + scratch_pages]
    scratch_positions = torch.tensor([page * block_size + pos for page in scratch_blocks
                                      for pos in range(block_size)], dtype=torch.int64)
    scratch_positions = scratch_positions[torch.randperm(scratch_positions.numel(), generator=generator)]
    node_cursor = 0
    for index, (prefix, count) in enumerate(zip(prefixes, counts)):
        needed = (prefix + block_size - 1) // block_size
        table = order[cursor:cursor + needed]
        cursor += needed
        positions = torch.arange(prefix, dtype=torch.int64)
        table_tensor = torch.tensor(table, dtype=torch.int64)
        logical = table_tensor[positions // block_size] * block_size + positions % block_size
        slots = scratch_positions[node_cursor:node_cursor + count]
        node_cursor += count
        parent = parents_for(count, seed + index)
        pages.append(table)
        prefix_slots.append(logical.to(device))
        node_slots.append(slots.to(device))
        parents.append(parent)
        masks.append(mask_from_parents(parent).to(device))
    metadata = PackedTreeMetadata.build(prefixes, pages, node_slots, masks, block_size)
    q = torch.randn((sum(counts), kv_heads * groups, head_dim), generator=generator).to(device=device, dtype=dtype)
    k = torch.full((total_pages, block_size, kv_heads, head_dim), float("nan"), dtype=dtype, device=device)
    v = torch.full_like(k, float("nan"))
    for prefix, tree in zip(prefix_slots, node_slots):
        live = torch.cat((prefix, tree))
        for pool in (k, v):
            values = torch.randn((live.numel(), kv_heads, head_dim), generator=generator).to(device=device, dtype=dtype)
            pool[live // block_size, live % block_size] = values
    return dict(q=q, k=k, v=v, metadata=metadata, prefix_slots=prefix_slots,
                node_slots=node_slots, parents=parents, groups=groups, scale=head_dim ** -.5)


def oracle_rows(fixture, output, *, all_rows=False):
    """CPU FP64 gates assembled from fixture-owned slots and parent indices."""
    import torch
    metadata = fixture["metadata"]
    q, k, v = (fixture[name].detach().cpu() for name in ("q", "k", "v"))
    output = output.detach().cpu()
    checks = []
    block = metadata.block_size
    for request, count in enumerate(metadata.node_counts_host):
        nodes = range(count) if all_rows else sorted({0, count // 2, count - 1})
        prefix_slots = fixture["prefix_slots"][request].detach().cpu()
        tree_slots = fixture["node_slots"][request].detach().cpu()
        for node in nodes:
            ancestors = parent_chain(fixture["parents"][request], node)
            slots = torch.cat((prefix_slots, tree_slots[ancestors]))
            oracle = fp64_attention(q[metadata.query_offsets[request] + node],
                k[slots // block, slots % block], v[slots // block, slots % block],
                fixture["scale"], fixture["groups"])
            check = numerical_metrics(output[metadata.query_offsets[request] + node], oracle)
            check.update(request=request, node=node, visible_keys=int(slots.numel()),
                         parent_chain=ancestors)
            frozen.require(check["fixed_bound_applicable"] and check["within_fixed_bound"],
                           f"FP32 parent-chain oracle failed: {check}")
            checks.append(check)
    return checks


def synthetic_case(fixture, *, query_tile=1, num_warps=4, all_rows=False, candidate=DEFAULT_CANDIDATE):
    import torch
    from nanovllm.speculative.jetspec.tree_attention import packed_tree_attention_reference
    args = [fixture[name] for name in ("q", "k", "v", "metadata", "scale", "groups")]
    reference = packed_tree_attention_reference(*args)
    actual = launch_candidate(args, candidate, query_tile=query_tile, num_warps=num_warps)
    fp32 = launch_candidate(args, candidate, output_dtype=torch.float32,
                           query_tile=query_tile, num_warps=num_warps)
    reference_fp32 = packed_tree_attention_reference(*args, output_dtype=torch.float32)
    repeat = launch_candidate(args, candidate, query_tile=query_tile, num_warps=num_warps)
    frozen.require(bool(torch.isfinite(actual).all()), "inaccessible NaN pages/slots were read")
    round_trip = bits_equal(fp32.to(actual.dtype), actual)
    reference_round_trip = bits_equal(reference_fp32.to(reference.dtype), reference)
    frozen.require(round_trip and reference_round_trip, "FP32 diagnostic does not reproduce production rounding")
    stable = bits_equal(actual, repeat)
    frozen.require(stable, "same-shape optimized kernel replay is not deterministic")
    delta = frozen.tensor_metrics(actual, reference, bound=BF16_BOUND)
    frozen.require(delta["passed"], "optimized attention exceeds the fixed BF16 envelope")
    return {"prefix_lengths": list(fixture["metadata"].prefix_lengths),
        "node_counts": list(fixture["metadata"].node_counts_host), "dtype": str(actual.dtype),
        "query_tile": query_tile, "num_warps": num_warps,
        "candidate": candidate_identity(candidate, query_tile=query_tile, num_warps=num_warps),
        "inactive_pages_are_nan": True, "production_vs_reference": delta,
        "production_vs_reference_bitwise_exact": bits_equal(actual, reference),
        "same_shape_repeat_bitwise_exact": stable,
        "fp32_output_round_trip_bitwise_exact": round_trip,
        "reference_fp32_output_round_trip_bitwise_exact": reference_round_trip,
        "candidate_vs_independent_fp64": oracle_rows(fixture, fp32, all_rows=all_rows),
        "reference_vs_independent_fp64": oracle_rows(fixture, reference_fp32, all_rows=all_rows)}


def isolation_case(fixture, *, query_tile=1, num_warps=4, candidate=DEFAULT_CANDIDATE):
    import torch
    metadata = fixture["metadata"]
    q, k, v = [fixture[name] for name in ("q", "k", "v")]
    def run(query, keys, values):
        return launch_candidate((query, keys, values, metadata, fixture["scale"], fixture["groups"]),
                                candidate, query_tile=query_tile, num_warps=num_warps)
    original = run(q, k, v)
    changed_q, changed_k, changed_v = q.clone(), k.clone(), v.clone()
    changed_q[metadata.request_slice(0)].mul_(7)
    slots = torch.cat((fixture["prefix_slots"][0], fixture["node_slots"][0]))
    block = metadata.block_size
    changed_k[slots // block, slots % block] *= -17
    changed_v[slots // block, slots % block] += 53
    changed = run(changed_q, changed_k, changed_v)
    neighbor = [bits_equal(original[metadata.request_slice(i)], changed[metadata.request_slice(i)])
                for i in range(1, len(metadata.prefix_lengths))]
    frozen.require(all(neighbor), "finite other-request operands contaminated a neighbor")
    chosen = metadata.node_counts_host[0] - 1
    path = parent_chain(fixture["parents"][0], chosen)
    off_branch = [node for node in range(metadata.node_counts_host[0]) if node not in path]
    changed_k, changed_v = k.clone(), v.clone()
    slots = fixture["node_slots"][0][off_branch]
    changed_k[slots // block, slots % block] *= 97
    changed_v[slots // block, slots % block] += 113
    off = run(q, changed_k, changed_v)
    path_equal = bits_equal(original[path], off[path])
    frozen.require(path_equal, "finite off-branch KV contaminated a chosen ancestor path")
    return {"other_request_neighbors_bitwise_exact": neighbor,
        "perturbed_request_changed": not bits_equal(original[metadata.request_slice(0)], changed[metadata.request_slice(0)]),
        "off_branch_ancestor_path_bitwise_exact": path_equal, "independent_parent_chain": path}


def run_synthetic(args, report):
    import torch
    # No model weights here. Mixed cases intentionally cross TILE64/page256.
    for dtype in (torch.bfloat16, torch.float32):
        for prefix in PREFIX_LENGTHS:
            fixture = synthetic_fixture((prefix, max(0, prefix - 1), prefix // 2), device="cuda", dtype=dtype)
            for tile in args.query_tiles:
                check = synthetic_case(fixture, query_tile=tile, num_warps=args.num_warps,
                                       all_rows=prefix <= 65, candidate=args.candidate)
                report["synthetic"].append(check)
                save_json(args.output, report)
                print(f"synthetic {dtype} P{prefix} tile{tile}: FP64/roundtrip/repeat PASS", flush=True)
    fixture = synthetic_fixture((65, 0, 257), device="cuda")
    report["synthetic_isolation"] = [isolation_case(fixture, query_tile=tile, num_warps=args.num_warps, candidate=args.candidate)
                                     for tile in args.query_tiles]


@contextmanager
def selected_kernel(query_tile, num_warps, *, reference=False, observe=None, candidate=DEFAULT_CANDIDATE):
    from nanovllm.speculative.jetspec import paged_backend, tree_attention
    def call(*args):
        if reference:
            output = tree_attention.packed_tree_attention_reference(*args)
        else:
            output = launch_candidate(args, candidate, query_tile=query_tile, num_warps=num_warps)
        if observe is not None:
            observe(args, output)
        return output
    with frozen.patch(paged_backend, "packed_tree_attention", call):
        yield


@contextmanager
def qualification_serving_modes(engine, args):
    """Freeze unchunked policy for historical fixed-first-Q408 controls.

    Phase-3.2's helper predates enable_chunked_prefill=True becoming a public
    default. Its eight-prompt workload totals 530 tokens: the new default
    512-token prefill budget leaves request eight partial and therefore makes
    the helper's required *first* Q408/16-controls assertion fail even for the
    reference kernel. Chunked serving is qualified separately and explicitly.
    """
    def set_mode(selected_engine, mode, draft, concurrency):
        frozen.require(selected_engine is engine and mode == "jetspec" and draft == args.draft,
                       "unexpected tree qualification mode switch")
        frozen.require(engine.is_finished(), "qualification configuration requires idle engine")
        engine.scheduler.max_num_seqs = concurrency
        engine.configure_jetspec(args.draft, optimization="serving", attention_backend="sdpa",
            enable_chunked_prefill=False, max_prefill_tokens=args.max_model_len,
            max_admissions_per_step=2)
        engine._jetspec_scheduler.max_num_seqs = concurrency
    with frozen.patch(serving, "set_mode", set_mode):
        yield


def record_and_require_trained_check(check, *, observed_layers, expected_layers,
                                     expected_controls, record_check=None):
    """Publish completed diagnostics BEFORE rejecting an unchanged hard gate.

    A failed whole-network envelope must remain a measured negative result,
    not an empty trained section plus an assertion string. The callback only
    materializes diagnostic evidence; it cannot alter gate thresholds.
    """
    check["gates"] = {
        "all_target_layers_executed": observed_layers == expected_layers and
            len(check["attention_operator_controls"]) == expected_controls,
        "full_network_fixed_bf16_envelope": all(check[key]["passed"] for key in
            ("all_layer_tree_kv", "final_hidden", "target_taps", "lm_head")),
        "argmax_flip_within_measured_envelope": check["argmax_flip_witness"]["all_flip_witnesses_consistent"],
        "canonical_history_unchanged": check["canonical_history_byte_exact"],
        "candidate_repeat_byte_exact": all(check["candidate_repeat_bitwise_exact"].values()),
    }
    check["per_layer_tree_kv_empirical_envelope_passed"] = all(row["metrics"]["passed"]
                                                               for row in check["tree_kv_by_layer"])
    check["passed"] = all(check["gates"].values())
    if record_check is not None:
        record_check(check)
    frozen.require(check["gates"]["all_target_layers_executed"], "not every Target layer executed the selected kernel")
    frozen.require(check["gates"]["full_network_fixed_bf16_envelope"], "trained full-network fixed BF16 envelope failed")
    frozen.require(check["gates"]["argmax_flip_within_measured_envelope"], "argmax flip exceeds measured same-state error envelope")
    frozen.require(check["gates"]["canonical_history_unchanged"] and
                   check["gates"]["candidate_repeat_byte_exact"], "trained replay modified canonical prefix or changed identical-state output")


def trained_same_state(runtime, requests, trees, transaction, metadata, verify, *, query_tile, num_warps,
                       candidate=DEFAULT_CANDIDATE, record_check=None):
    """Replay BOTH entire Target forwards before acceptance/physical commit.

    Each candidate layer's raw BF16 operands are checked against reference
    attention and an FP64 parent-chain oracle. Selected layers are reported;
    this is diagnostic and includes deliberate CPU copies/synchronizations.
    """
    import torch
    from nanovllm.speculative.jetspec import tree_attention
    layer_count = int(runtime.kv_pool.shape[1])
    selected_layers = {0, layer_count // 2, layer_count - 1}
    block = runtime.block_size
    tree_slots = torch.cat(transaction.node_slots)
    prefix_history = [chunks.raw_kv(runtime, r.state.logical_slots).clone() for r in requests]
    final_hidden = []
    hook = runtime.target.lm_head.register_forward_pre_hook(
        lambda _module, inputs: final_hidden.append(inputs[0].detach().cpu().clone()))
    try:
        with selected_kernel(query_tile, num_warps, reference=True):
            logits_ref, taps_ref = verify(requests, trees, transaction, metadata)
    finally:
        hook.remove()
    frozen.require(len(final_hidden) == 1, "reference final hidden was not observed")
    kv_ref = chunks.raw_kv(runtime, tree_slots).clone()
    layer, controls = 0, []
    def observe(operands, actual):
        nonlocal layer
        current = layer
        layer += 1
        if current not in selected_layers:
            return
        q, k, v, meta, scale, groups = operands
        fp32 = launch_candidate(operands, candidate, output_dtype=torch.float32,
                                query_tile=query_tile, num_warps=num_warps)
        reference_fp32 = tree_attention.packed_tree_attention_reference(*operands, output_dtype=torch.float32)
        reference = tree_attention.packed_tree_attention_reference(*operands)
        frozen.require(bits_equal(fp32.to(actual.dtype), actual), "trained optimized attention failed FP32 round-trip")
        frozen.require(bits_equal(reference_fp32.to(reference.dtype), reference), "trained reference attention failed FP32 round-trip")
        prefix_slots, parents = [], []
        for request, tree in zip(requests, trees):
            # NOT metadata.block_tables / metadata.qq_bias: reconstruct from
            # canonical owner state plus independent tree parent indices.
            positions = torch.arange(request.state.cache_len, device=q.device)
            table = torch.tensor(request.state.owned_blocks, dtype=torch.int64, device=q.device)
            prefix_slots.append(table[positions // block] * block + positions % block)
            parents.append(getattr(tree, "host_parents", None) or tree.parent_indices.detach().cpu().tolist())
        fixture = dict(q=q, k=k, v=v, metadata=meta, scale=scale, groups=groups,
                       prefix_slots=prefix_slots, node_slots=transaction.node_slots, parents=parents)
        controls.append({"layer": current, "input_dtype": str(q.dtype),
            "output_round_trip_bitwise_exact": True,
            "same_operands_vs_reference": frozen.tensor_metrics(actual, reference, bound=BF16_BOUND),
            "candidate_vs_cpu_fp64": oracle_rows(fixture, fp32),
            "reference_vs_cpu_fp64": oracle_rows(fixture, reference_fp32)})
        frozen.require(controls[-1]["same_operands_vs_reference"]["passed"], "trained operator BF16 envelope failed")
    hook = runtime.target.lm_head.register_forward_pre_hook(
        lambda _module, inputs: final_hidden.append(inputs[0].detach().cpu().clone()))
    try:
        with selected_kernel(query_tile, num_warps, observe=observe, candidate=candidate):
            logits, taps = verify(requests, trees, transaction, metadata)
    finally:
        hook.remove()
    frozen.require(len(final_hidden) == 2, "candidate final hidden was not observed")
    kv = chunks.raw_kv(runtime, tree_slots).clone()
    with selected_kernel(query_tile, num_warps, candidate=candidate):
        repeated_logits, repeated_taps = verify(requests, trees, transaction, metadata)
    repeat_kv = chunks.raw_kv(runtime, tree_slots)
    history = all(bits_equal(prior, chunks.raw_kv(runtime, request.state.logical_slots))
                  for prior, request in zip(prefix_history, requests))
    check = {"prefix_lengths": list(metadata.prefix_lengths), "node_counts": list(metadata.node_counts_host),
        "query_tile": query_tile, "candidate": candidate_identity(candidate, query_tile=query_tile, num_warps=num_warps),
        "attention_operator_controls": controls,
        "all_layer_tree_kv": frozen.tensor_metrics(kv, kv_ref, bound=BF16_BOUND),
        "tree_kv_by_layer": [{"layer": index, "metrics": frozen.tensor_metrics(
            kv[:, index], kv_ref[:, index], bound=BF16_BOUND)} for index in range(layer_count)],
        "final_hidden": frozen.tensor_metrics(final_hidden[1], final_hidden[0], bound=BF16_BOUND),
        "target_taps": frozen.tensor_metrics(taps, taps_ref, bound=BF16_BOUND),
        "lm_head": frozen.tensor_metrics(logits, logits_ref, bound=BF16_BOUND),
        "argmax_flip_witness": argmax_flip_witness(logits.detach().cpu(), logits_ref.detach().cpu()),
        "candidate_repeat_bitwise_exact": {"tree_kv": bits_equal(kv, repeat_kv),
            "taps": bits_equal(taps, repeated_taps), "logits": bits_equal(logits, repeated_logits)},
        "canonical_history_byte_exact": history,
        "reference_vs_candidate_bitwise": {"tree_kv": bits_equal(kv, kv_ref),
            "taps": bits_equal(taps, taps_ref), "logits": bits_equal(logits, logits_ref)}}
    record_and_require_trained_check(check, observed_layers=layer, expected_layers=layer_count,
                                     expected_controls=len(selected_layers), record_check=record_check)
    return (logits, taps), check


def trained_qualification(args, report):
    import torch
    from nanovllm import LLM
    engine = None
    try:
        engine = LLM(args.target, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=.8,
                     max_num_seqs=8, max_model_len=args.max_model_len,
                     max_num_batched_tokens=args.max_model_len, kvcache_block_size=256)
        engine.configure_jetspec(args.draft, optimization="serving", attention_backend="sdpa",
            enable_chunked_prefill=False, max_admissions_per_step=8)
        runtime = engine._jetspec_scheduler.runtime
        base_prompt = runtime.tokenizer.encode("Explain why a tree has independent ancestor paths.", add_special_tokens=False)
        lengths = (33, 257, 1024, 2048, 65, 128, 255, 512)
        for tile in args.query_tiles:
            requests = []
            original_verify = runtime._verify_batch
            checks = []
            round_report = {"query_tile": tile, "rounds": checks, "raw_commit_checks": None}
            report["trained_same_state"].append(round_report)
            save_json(args.output, report)
            def record_check(check):
                checks.append(check)
                save_json(args.output, report)
            def verify(requests, trees, transaction, metadata):
                result, check = trained_same_state(runtime, requests, trees, transaction, metadata,
                    original_verify, query_tile=tile, num_warps=args.num_warps, candidate=args.candidate,
                    record_check=record_check)
                return result
            try:
                requests = [runtime.create_request((base_prompt * ((n + len(base_prompt) - 1) // len(base_prompt)))[:n],
                    max_new_tokens=32, tree_budget=(63, 31, 47)[i % 3], ignore_eos=True,
                    request_id=f"treequal-{tile}-{i}") for i, n in enumerate(lengths)]
                with frozen.patch(runtime, "_verify_batch", verify), serving.ServingProbe(engine, diagnostic=True) as probe:
                    for _ in range(2):
                        runtime.step(requests)
                    raw_checks = probe.metrics()
                round_report["raw_commit_checks"] = raw_checks
                save_json(args.output, report)
                frozen.require(all(raw_checks["all_layer_raw_copy_checks"]) and all(raw_checks["history_and_rejected_checks"]),
                               "trained optimized accepted-only commit or history gate failed")
            finally:
                for request in requests:
                    if runtime.requests.get(request.request_id) is request:
                        runtime.cancel(request)
                runtime.release_idle_scratch()
        prompts = {row["id"]: row for row in json.loads(Path(args.oracle).read_text())["prompts"]}
        with selected_kernel(args.query_tiles[-1], args.num_warps, candidate=args.candidate), \
                qualification_serving_modes(engine, args):
            report["packed_request_isolation"] = serving.eight_request_isolation(engine, prompts, args.draft)
            isolation = report["packed_request_isolation"]
            frozen.require(isolation["real_eight_request_packed_shape"] and isolation["finite_same_shape_controls_passed"],
                           "trained cross-request/off-branch isolation failed")
            frozen.require(all(isolation["checks"]["all_layer_raw_copy_checks"]) and
                           all(isolation["checks"]["history_and_rejected_checks"]), "packed isolation raw commit/history failed")
            serving.set_mode(engine, "jetspec", args.draft, 2)
            with serving.ServingProbe(engine, diagnostic=True) as probe:
                first = serving.serving_run(engine, serving.qualification_workload(prompts), mode="jetspec",
                    clock="step", label="treekernel-lifecycle-a", max_wall_s=args.deadline)
                raw_checks = probe.metrics()
            second = serving.serving_run(engine, serving.qualification_workload(prompts), mode="jetspec",
                clock="step", label="treekernel-lifecycle-b", max_wall_s=args.deadline)
            exact = serving.replay_signature(first) == serving.replay_signature(second)
            report["lifecycle"] = dict(first=first, second=second, raw_checks=raw_checks,
                                       fixed_schedule_replay_exact=exact)
            report["lifecycle"]["semantic_gates"] = lifecycle_semantics(first, runtime.eos_token_ids)
            frozen.require(exact and first["dynamic_live_arrival_seen"], "dynamic lifecycle/exactly-once replay failed")
            frozen.require(all(raw_checks["all_layer_raw_copy_checks"]) and all(raw_checks["history_and_rejected_checks"]),
                           "dynamic lifecycle accepted-only commit/history failed")
            report["allocator_pressure"] = serving.allocator_pressure(engine, prompts)
            report["preemption_recompute"] = serving.preemption_qualification(engine, prompts)
            frozen.require(all(report["allocator_pressure"][key] for key in ("deferred_seen", "recovered")), "allocator pressure did not recover")
            frozen.require(all(report["preemption_recompute"][key] for key in
                ("preemption_seen", "resume_seen", "output_exactly_once", "fixed_schedule_replay_exact", "scratch_one_page_or_less")),
                "preemption/recompute state gate failed")
            report["chunked_lifecycle_recompute"] = chunks.lifecycle_qualification(engine, args)
        engine.disable_jetspec()
        report["allocator_cleanup"] = not engine.scheduler.block_manager.used_block_ids
        frozen.require(report["allocator_cleanup"], "trained qualification leaked canonical/scratch pages")
    finally:
        if engine is not None:
            scheduler = getattr(engine, "_jetspec_scheduler", None)
            if scheduler is not None:
                for request_id in list(scheduler.requests):
                    engine.cancel_request(request_id)
                for request in list(scheduler.runtime.requests.values()):
                    scheduler.runtime.cancel(request)
                for context in list(scheduler.runtime.prefills.values()):
                    scheduler.runtime.cancel_prefill(context)
                scheduler.runtime.release_idle_scratch()
                scheduler.drain_events()
            engine.exit()


def run(args):
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    import nanovllm
    torch.set_num_threads(1)
    torch.manual_seed(317)
    frozen.require(Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()), "wrong selected production import")
    source = frozen.qualification_identity(args)
    script = Path(__file__).resolve()
    report = {"kind": "untimed Phase-5 tree-kernel qualification", "status": "running", "passed": False,
        "source": source, "tree_qualification_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "tree_qualification_helper_sha256": {str(Path(module.__file__).resolve()):
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in (frozen, serving, chunks)},
        "fp32_pre_bf16_scaled_max_and_relative_rms_bound": FP32_ATTENTION_BOUND,
        "full_network_bf16_scaled_max_and_relative_rms_bound": BF16_BOUND,
        "cross_shape_token_bitwise_required": False,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name()},
        "kernel_options": {"query_tiles": args.query_tiles, "num_warps": args.num_warps},
        "candidate": candidate_identity(args.candidate, query_tile=args.query_tiles[-1], num_warps=args.num_warps),
        "synthetic": [], "trained_same_state": []}
    save_json(args.output, report)
    try:
        run_synthetic(args, report)
        if not args.synthetic_only:
            trained_qualification(args, report)
        report["source_unchanged"] = frozen.qualification_identity(args)["production_source_sha256"] == source["production_source_sha256"]
        frozen.require(report["source_unchanged"], "production source changed during qualification")
        report.update(status="complete", passed=True, all_gates_passed=True)
        save_json(args.output, report)
        print("ALL tree-kernel qualification gates PASS", flush=True)
    except BaseException as error:
        report.update(status="failed", passed=False, error=f"{type(error).__name__}: {error}")
        save_json(args.output, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--target", default=serving.TARGET)
    parser.add_argument("--draft", default=serving.DRAFT)
    parser.add_argument("--oracle", default=serving.ORACLE)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--deadline", type=float, default=600)
    parser.add_argument("--query-tiles", default="1,2")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE,
                        help="Explicit module:function, e.g. tree_prefix:packed_tree_attention_prefix")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.query_tiles = [int(tile) for tile in args.query_tiles.split(",")]
    if not args.query_tiles or any(tile not in (1, 2, 4) for tile in args.query_tiles):
        parser.error("query tiles must be a nonempty subset of 1,2,4")
    run(args)


if __name__ == "__main__":
    main()
