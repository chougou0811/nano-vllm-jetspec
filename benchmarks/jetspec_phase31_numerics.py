"""Causal, identical-state qualification of the six Phase 3.1 AR mismatches.

Run in a fresh idle TP1 process. No production forwards or numerical flags are
changed. Real forwards are traced; reference AR is teacher-forced from the SAME
round-start committed tokens/KV, not from an already numerically drifted cache.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jetspec_phase31 import identity, first_divergence, TARGET, DRAFT, ORACLE
from jetspec_phase3 import save_json
from jetspec_layer_trace import LayerTrace, compare_traces, tensor_metrics, logit_margin, replay_packed_attention_fp32
from jetspec_numeric_oracles import parent_chain, fp64_attention, numerical_metrics, argmax_flip_witness
from nanovllm import LLM
from nanovllm.speculative.jetspec.batch_runtime import JetSpecBatchRuntime
from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata
from nanovllm.speculative.jetspec.state import PagedTargetState, BatchTreeTransaction


def cpu(x):
    return x.detach().cpu().clone()


def read_slots(pool, slots):
    return pool[:, :, slots // 256, slots % 256]


def tree_copy(tree, reference=False):
    from jetspec.tree import build_ancestor_matrix
    return {"token_ids": cpu(tree.token_ids), "depth": cpu(tree.depth),
            "parent_indices": None if reference else cpu(tree.parent_indices),
            "ancestor": cpu(build_ancestor_matrix(tree).bool()), "num_nodes": int(tree.num_nodes)}


def device_trees(snapshot, device):
    return [SimpleNamespace(**{k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in tree.items()}) for tree in snapshot["trees"]]


class Capture:
    def __init__(self, runner):
        self.runner = runner
        self.original_verify, self.original_create = runner._verify_batch, runner.create_request
        self.snapshots, self.prompt_ids, self.reference = [], [], False
        self.cursor = 0

        def create(*args, **kwargs):
            request = self.original_create(*args, **kwargs)
            request.prompt_id = self.prompt_ids[self.cursor]
            self.cursor += 1
            return request

        def verify(requests, trees, tx, meta):
            result = self.original_verify(requests, trees, tx, meta)
            from jetspec.tree import gpu_tree_accept
            snapshot = {"requests": [], "trees": [tree_copy(t, self.reference) for t in trees],
                        "paths": {}, "path_logits": {}, "query_offsets": list(meta.query_offsets)}
            for i, (request, tree) in enumerate(zip(requests, trees)):
                state = request.state
                snapshot["requests"].append({"prompt_id": request.prompt_id,
                    "request_id": request.request_id, "output_ids": list(request.output_ids),
                    "committed": cpu(state.committed), "hidden": cpu(state.target_hidden),
                    "kv": cpu(read_slots(state.kv_pool, state.logical_slots)),
                    "cache_len": state.cache_len})
                rows = meta.request_slice(i)
                if self.reference:
                    path = torch.zeros(1, dtype=torch.long, device=result[0].device)
                else:
                    path, _, _ = gpu_tree_accept(tree.token_ids, result[0][rows].argmax(-1),
                        tree.parent_indices, tree.depth, max_depth=self.runner.tree_depth)
                snapshot["paths"][request.prompt_id] = path.tolist()
                snapshot["path_logits"][request.prompt_id] = cpu(result[0][rows].index_select(0, path.long()))
            self.snapshots.append(snapshot)
            return result

        runner.create_request, runner._verify_batch = create, verify

    def generate(self, prompt_ids, prompts, budgets, *, reference=False):
        self.snapshots, self.cursor, self.prompt_ids = [], 0, list(prompt_ids)
        self.reference = reference
        method = self.runner.generate_target_batch if reference else self.runner.generate_batch
        result = method(prompts, tree_budgets=budgets, max_new_tokens=32)
        return result, self.snapshots

    def restore(self):
        self.runner.create_request, self.runner._verify_batch = self.original_create, self.original_verify


class Replay:
    def __init__(self, runner, snapshot):
        self.runner, self.snapshot = runner, snapshot
        self.requests = []
        self.tx = None
        try:
            for request in snapshot["requests"]:
                kv = request["kv"].to(runner.kv_pool.device)
                pairs = [(kv[0, layer], kv[1, layer]) for layer in range(kv.shape[1])]
                state = PagedTargetState.from_prefill(request["committed"].to(kv.device), pairs,
                    request["hidden"].to(kv.device), runner.kv_pool, runner.block_manager, 256)
                if not torch.equal(cpu(read_slots(runner.kv_pool, state.logical_slots)), request["kv"]):
                    raise AssertionError("snapshot restoration changed canonical KV bytes")
                self.requests.append(SimpleNamespace(state=state, request_id=request["request_id"],
                                                       prompt_id=request["prompt_id"]))
        except BaseException:
            self.close()
            raise

    def admit(self, trees):
        if self.tx is not None:
            self.tx.abort()
        self.tx = BatchTreeTransaction.admit([r.state for r in self.requests],
            [t.num_nodes for t in trees], [min(t.num_nodes, 16) for t in trees], self.runner.arena)
        self.meta = PackedTreeMetadata.build([r.state.cache_len for r in self.requests],
            [r.state.owned_blocks for r in self.requests], self.tx.node_slots,
            [t.ancestor for t in trees], 256, request_ids=[r.request_id for r in self.requests])
        return self.meta

    def forward(self, trees, selected, *, full_layers=()):
        meta = self.admit(trees)
        with LayerTrace(self.runner.target, selected, capture_full_layers=full_layers) as trace:
            logits, taps = self.runner._verify_batch(self.requests, trees, self.tx, meta)
        return trace, logits, taps

    def close(self):
        if self.tx is not None:
            self.tx.abort()
        for request in self.requests:
            request.state.clear()
        self.requests = []


def row_trace(trace, index):
    return SimpleNamespace(records=OrderedDict((k, v[index:index + 1]) for k, v in trace.records.items()),
        stage_inputs={k: v[index:index + 1] for k, v in trace.stage_inputs.items()},
        operator_shapes=trace.operator_shapes, selected_rows=(trace.selected_rows[index],),
        full_records=getattr(trace, "full_records", {}))


def compare_path(packed, serial):
    rows = [compare_traces(row_trace(packed, i), trace, include_equal=False)
            for i, trace in enumerate(serial)]
    if any(not row["trace_schema_equal"] for row in rows):
        raise AssertionError("packed and serial traces have different stage schemas")
    first = next(({"path_offset": i, **row["earliest_divergence"]}
                  for i, row in enumerate(rows) if row["earliest_divergence"] is not None), None)
    return {"earliest_chronological_node": first, "by_path_offset": rows}


def root_trees(replay, budgets):
    device = replay.runner.kv_pool.device
    return [SimpleNamespace(token_ids=torch.cat((r.state.committed[0, -1:],
                torch.zeros(n - 1, dtype=torch.long, device=device))),
            depth=torch.cat((torch.zeros(1, dtype=torch.long, device=device),
                             torch.ones(n - 1, dtype=torch.long, device=device))),
            num_nodes=n, ancestor=torch.eye(n, dtype=torch.bool, device=device))
            for r, n in zip(replay.requests, budgets)]


def serial_from_same_state(runner, snapshot, chosen, path, *, full_layers=(), capture_prefix_states=False):
    replay = Replay(runner, snapshot)
    traces, final_state, earlier_states = [], None, []
    source = snapshot["trees"][chosen]["token_ids"][path]
    budgets = [t["num_nodes"] for t in snapshot["trees"]]
    try:
        for step in range(len(path)):
            trees = root_trees(replay, budgets)
            if int(trees[chosen].token_ids[0]) != int(source[step]):
                raise AssertionError("teacher-forced AR anchor differs from packed path token")
            row = sum(budgets[:chosen])
            trace, logits, taps = replay.forward(trees, [row], full_layers=full_layers)
            traces.append(trace)
            if capture_prefix_states or step == len(path) - 1:
                current_state = {"prefix_kv": cpu(read_slots(runner.kv_pool, replay.requests[chosen].state.logical_slots)),
                    "prefix_len": replay.requests[chosen].state.cache_len,
                    "tree_roots": [int(t.token_ids[0]) for t in trees], "query_offsets": list(replay.meta.query_offsets)}
                if step == len(path) - 1:
                    final_state = current_state
                    if capture_prefix_states:
                        final_state["earlier_states"] = earlier_states
                else:
                    earlier_states.append(current_state)
            if step == len(path) - 1:
                break
            tokens, features, paths = [], [], []
            old = []
            for i, request in enumerate(replay.requests):
                lo, hi = replay.meta.query_offsets[i:i + 2]
                correction = source[step + 1].to(logits.device) if i == chosen else logits[lo].argmax()
                tokens.append(torch.cat((request.state.committed, correction.reshape(1, 1)), 1))
                features.append(taps[lo:hi].unsqueeze(0))
                paths.append(torch.zeros(1, dtype=torch.long, device=logits.device))
                old.append(cpu(read_slots(runner.kv_pool, request.state.logical_slots)))
            src = torch.cat([slots[:1] for slots in replay.tx.node_slots])
            payload = cpu(read_slots(runner.kv_pool, src))
            replay.tx.commit(features, paths, tokens)
            dst = torch.cat([request.state.logical_slots[-1:] for request in replay.requests])
            if not torch.equal(payload, cpu(read_slots(runner.kv_pool, dst))):
                raise AssertionError("teacher-forced AR commit changed raw KV")
            for request, history in zip(replay.requests, old):
                actual = cpu(read_slots(runner.kv_pool, request.state.logical_slots[:-1]))
                if not torch.equal(actual, history):
                    raise AssertionError("teacher-forced AR commit changed historical KV")
                request.state.assert_round_invariant()
    finally:
        replay.close()
    return traces, final_state


def packed_trace(runner, snapshot, chosen, path, *, full_layers=()):
    replay = Replay(runner, snapshot)
    trees = device_trees(snapshot, runner.kv_pool.device)
    rows = [snapshot["query_offsets"][chosen] + node for node in path]
    try:
        trace, logits, _ = replay.forward(trees, rows, full_layers=full_layers)
        expected_path = snapshot["paths"][snapshot["requests"][chosen]["prompt_id"]]
        expected_logits = snapshot["path_logits"][snapshot["requests"][chosen]["prompt_id"]]
        expected = expected_logits[[expected_path.index(node) for node in path]]
        if not torch.equal(trace.records["lm_head"], expected):
            raise AssertionError("tracing or restoring physical pages changed original packed logits")
        return trace
    finally:
        replay.close()


def semantic_checks(snapshot):
    for request, tree in zip(snapshot["requests"], snapshot["trees"]):
        if tree["parent_indices"] is None:
            continue
        for node in range(tree["num_nodes"]):
            chain = parent_chain(tree["parent_indices"], node)
            visible = tree["ancestor"][node].nonzero().flatten().tolist()
            if chain != visible or len(chain) - 1 != int(tree["depth"][node]):
                raise AssertionError("ancestor visibility or logical depth differs from independent parent chain")
        if request["committed"].shape[1] - 1 != request["cache_len"]:
            raise AssertionError("uncached-anchor state mismatch")
    return True


def isolation_control(runner, snapshot, chosen, path, base):
    replay = Replay(runner, snapshot)
    device = runner.kv_pool.device
    trees = device_trees(snapshot, device)
    rows = [snapshot["query_offsets"][chosen] + node for node in path]
    try:
        changed = copy.deepcopy(trees)
        for index, tree in enumerate(changed):
            if index != chosen:
                tree.token_ids.zero_()
        other, _, _ = replay.forward(changed, rows)
        request_equal = compare_traces(base, other)["all_compared_stages_bitwise_equal"]
        changed = copy.deepcopy(trees)
        off = torch.ones(changed[chosen].num_nodes, dtype=torch.bool, device=device)
        off[path] = False
        changed[chosen].token_ids[off] = 0
        changed[chosen].ancestor[off] = False
        off_ids = off.nonzero().flatten()
        changed[chosen].ancestor[off_ids, off_ids] = True
        branch, _, _ = replay.forward(changed, rows)
        branch_equal = compare_traces(base, branch)["all_compared_stages_bitwise_equal"]
        if not request_equal or not branch_equal:
            raise AssertionError("fixed-layout per-layer isolation failed")
        return {"other_request_all_stages_bitwise_equal": request_equal,
                "off_branch_all_stages_bitwise_equal": branch_equal}
    finally:
        replay.close()


def attention_origin(runner, snapshot, chosen, path, packed, serial, serial_state, layer):
    """Same operands, independently compacted visible keys, actual Triton replay."""
    prefix = snapshot["requests"][chosen]["kv"]
    layer_obj = runner.target.model.layers[layer].self_attn
    key = f"layer_{layer:02d}."
    q = packed.full_records[key + "q_rope"]
    k = packed.full_records[key + "k_rope"]
    v = packed.full_records[key + "v_scattered"]
    local_node = path[-1]
    global_row = snapshot["query_offsets"][chosen] + local_node
    ancestors = parent_chain(snapshot["trees"][chosen]["parent_indices"], local_node)
    tree_start = snapshot["query_offsets"][chosen]
    node_ids = torch.tensor([tree_start + n for n in ancestors], dtype=torch.long)
    visible_k = torch.cat((prefix[0, layer], k[node_ids]), 0)
    visible_v = torch.cat((prefix[1, layer], v[node_ids]), 0)
    oracle = fp64_attention(q[global_row], visible_k, visible_v,
                           layer_obj.scaling, layer_obj.num_heads // layer_obj.num_kv_heads)
    replay = Replay(runner, snapshot)
    trees = device_trees(snapshot, runner.kv_pool.device)
    try:
        meta = replay.admit(trees)
        device = runner.kv_pool.device
        kp, vp = runner.kv_pool[0, layer], runner.kv_pool[1, layer]
        slots = meta.tree_slots
        kp[slots // 256, slots % 256] = k.to(device)
        vp[slots // 256, slots % 256] = v.to(device)
        out = replay_packed_attention_fp32(q.to(device), kp, vp, meta,
                    layer_obj.scaling, layer_obj.num_heads // layer_obj.num_kv_heads)
        sparse = cpu(out[global_row])
        stored = packed.records[key + "attention_output"][-1]
        rounded_equal = torch.equal(sparse.to(torch.bfloat16), stored)
        # Construct a one-query canonical-prefix+current-key reference. Prefix
        # includes ancestors BEFORE self, all copied from the SAME sparse run.
        n_prefix = visible_k.shape[0] - 1
        pages = (n_prefix + 255) // 256
        ck = torch.zeros(pages + 1, 256, kp.shape[2], kp.shape[3], dtype=kp.dtype, device=device)
        cv = torch.zeros_like(ck)
        positions = torch.arange(n_prefix, device=device)
        ck[positions // 256, positions % 256] = visible_k[:-1].to(device)
        cv[positions // 256, positions % 256] = visible_v[:-1].to(device)
        ck[pages, 0], cv[pages, 0] = visible_k[-1].to(device), visible_v[-1].to(device)
        compact_meta = PackedTreeMetadata.build([n_prefix], [list(range(pages))],
            [torch.tensor([pages * 256], device=device)], [torch.ones(1, 1, dtype=torch.bool, device=device)], 256)
        compact = cpu(replay_packed_attention_fp32(q[global_row:global_row + 1].to(device), ck, cv,
            compact_meta, layer_obj.scaling, layer_obj.num_heads // layer_obj.num_kv_heads)[0])
        sparse_error, compact_error = numerical_metrics(sparse, oracle), numerical_metrics(compact, oracle)
        if not rounded_equal or not sparse_error["within_fixed_bound"] or not compact_error["within_fixed_bound"]:
            raise AssertionError("attention pre-round oracle/round-trip gate failed")
        same_q = torch.equal(packed.records[key + "q_rope"][-1], serial.records[key + "q_rope"][0])
        serial_k = torch.cat((serial_state["prefix_kv"][0, layer], serial.records[key + "k_rope"]), 0)
        serial_v = torch.cat((serial_state["prefix_kv"][1, layer], serial.records[key + "v_scattered"]), 0)
        same_k, same_v = torch.equal(visible_k, serial_k), torch.equal(visible_v, serial_v)
        if not same_q or not same_k or not same_v:
            raise AssertionError("first attention origin does not have identical mathematical operands")
        serial_output = serial.records[key + "attention_output"][0]
        serial_roundtrip = torch.equal(compact.to(torch.bfloat16), serial_output)
        if not serial_roundtrip:
            raise AssertionError("compact attention control did not reproduce actual serial AR output")
        report = {"visible_key_count": len(visible_k), "independent_ancestor_indices": ancestors,
            "original_bf16_equals_fp32_replay_rounded": rounded_equal,
            "actual_serial_bf16_equals_compact_fp32_rounded": serial_roundtrip,
            "same_q_as_serial": same_q, "same_visible_k_as_serial": same_k, "same_visible_v_as_serial": same_v,
            "sparse_vs_compact_fp32": tensor_metrics(sparse, compact),
            "sparse_vs_fp64": sparse_error, "compact_vs_fp64": compact_error,
            "sparse_vs_compact_bf16": tensor_metrics(sparse.to(torch.bfloat16), compact.to(torch.bfloat16)),
            "same_visible_kv_constructed_from_raw_slots": True,
            "fp64_nearest_bf16_vs_original": numerical_metrics(stored, oracle, bf16_output=True)}
        return report
    finally:
        replay.close()


def shape_linear_control(runner, packed, reference, chosen_offset):
    """Replay actual split Q projection with identical chosen input, two M's."""
    stage = "layer_00.q_raw"
    a = packed.full_records[stage + "_input"]
    b = reference.full_records[stage + "_input"]
    row_a, row_b = packed.selected_rows[chosen_offset], reference.selected_rows[0]
    if not torch.equal(a[row_a], b[row_b]):
        raise AssertionError("shape-only GEMM control operands differ")
    layer = runner.target.model.layers[0].self_attn
    weight = layer.qkv_proj.weight[:layer.q_size]
    bias = None if layer.qkv_proj.bias is None else layer.qkv_proj.bias[:layer.q_size]
    import torch.nn.functional as F
    left = cpu(F.linear(a.to(weight.device), weight, bias)[row_a])
    right = cpu(F.linear(b.to(weight.device), weight, bias)[row_b])
    if not torch.equal(left, packed.records[stage][chosen_offset]) or not torch.equal(right, reference.records[stage][0]):
        raise AssertionError("isolated vanilla F.linear failed to reproduce QKV shape drift")
    oracle = cpu(F.linear(a[row_a:row_a + 1].to(weight.device).double(), weight.double(),
                         None if bias is None else bias.double())[0])
    le, re = numerical_metrics(left, oracle, bf16_output=True), numerical_metrics(right, oracle, bf16_output=True)
    # A separate, predeclared BF16 projection envelope, NOT the FP32 attention
    # bound and NOT an arbitrary logit tolerance: two BF16 unit roundoffs.
    bound = 2 ** -7
    passed = all(e["max_abs_error"] <= bound * max(1.0, e["reference_absmax"])
                 and e["relative_rms_error"] <= bound for e in (le, re))
    if not passed:
        raise AssertionError("same-input BF16 GEMM exceeded fixed two-unit-roundoff envelope")
    return {"stage": stage, "same_selected_input_bitwise": True,
        "packed_shape": list(a.shape), "reference_shape": list(b.shape),
        "vanilla_F_linear_reproduces_both_actual_outputs": True,
        "actual_delta": tensor_metrics(left, right), "packed_vs_fp64": le,
        "reference_vs_fp64": re, "bf16_projection_scaled_maxabs_and_relative_rms_bound": bound,
        "both_within_fixed_bf16_projection_envelope": passed}


def locate(snapshots, prompt_id, output_index):
    for round_id, snapshot in enumerate(snapshots):
        for chosen, request in enumerate(snapshot["requests"]):
            if request["prompt_id"] != prompt_id:
                continue
            for path_offset, node in enumerate(snapshot["paths"][prompt_id]):
                if len(request["output_ids"]) + int(snapshot["trees"][chosen]["depth"][node]) == output_index:
                    return round_id, snapshot, chosen, path_offset
    raise AssertionError(f"could not locate {prompt_id} token {output_index}")


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    source = identity(args)
    diagnostic_names = ("jetspec_phase31_numerics.py", "jetspec_layer_trace.py", "jetspec_numeric_oracles.py")
    diagnostic_start = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                        for name in diagnostic_names}
    initial = json.loads(Path(args.qualification).read_text())
    failures = initial["phase31_correctness"]["strict_root_ar_failures"]
    previous = json.loads(Path(args.original).read_text())
    cases = {}
    for failure in failures:
        case = failure["case"]
        sample = next(s for s in previous["samples"] if s["case"] == case and s["mode"] == "packed_jetspec")
        entry = next(c for c in previous["correctness"] if c["case"] == case)
        cases[case] = {"prompt_ids": [c["prompt_id"] for c in entry["checks"]], "budgets": sample["tree_budgets"]}
    by_prompt = {p["id"]: p for p in json.loads(Path(args.oracle).read_text())["prompts"]}
    engine = LLM(args.target, enforce_eager=True, tensor_parallel_size=1,
        gpu_memory_utilization=0.8, max_num_batched_tokens=4096, max_model_len=4096,
        max_num_seqs=4, kvcache_block_size=256)
    runner = JetSpecBatchRuntime(engine.model_runner.model, engine.tokenizer, args.draft,
        kv_pool=engine.model_runner.kv_cache, block_manager=engine.scheduler.block_manager)
    capture = Capture(runner)
    report = {"schema_version": 1, "kind": "phase31_identical_state_numerical_causality", **source,
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32},
        "cases": [], "notes": ["layer numbers and token indices are zero based",
            "serial reference teacher-forces identical ancestor tokens from bitwise identical round-start KV",
            "original closed loops are separately traced; equal emitted tokens do not imply equal cached KV",
            "independent parent-chain FP64 oracle uses actual q/k/v, not the Triton mask implementation",
            "FP32 attention fixed oracle bounds 1e-4 scaled maxabs and relativeRMS; BF16 rounding separate",
            "no production algorithm or precision flag is changed"]}
    try:
        for case, config in cases.items():
            inputs = [by_prompt[name]["prompt_token_ids"] for name in config["prompt_ids"]]
            packed_result, packed_snapshots = capture.generate(config["prompt_ids"], inputs, config["budgets"])
            ar_result, ar_snapshots = capture.generate(config["prompt_ids"], inputs, config["budgets"], reference=True)
            capture.restore()
            for failure in [f for f in failures if f["case"] == case]:
                prompt = failure["prompt_id"]
                output_index = failure["first_divergence"]["index"]
                prompt_index = config["prompt_ids"].index(prompt)
                actual_tokens = packed_result["requests"][prompt_index]["token_ids"]
                reference_tokens = ar_result["requests"][prompt_index]["token_ids"]
                if first_divergence(actual_tokens, reference_tokens) != failure["first_divergence"]:
                    raise AssertionError("current closed loops do not reproduce the full original common token prefix")
                round_id, snapshot, chosen, offset = locate(packed_snapshots, prompt, output_index)
                _, ar_snapshot, ar_chosen, _ = locate(ar_snapshots, prompt, output_index)
                path = snapshot["paths"][prompt][:offset + 1]
                pt = packed_trace(runner, snapshot, chosen, path, full_layers=(0,))
                at = packed_trace(runner, ar_snapshot, ar_chosen, [0], full_layers=(0,))
                original_compare = compare_traces(row_trace(pt, offset), at, include_equal=False)
                original_delta = argmax_flip_witness(pt.records["lm_head"][-1], at.records["lm_head"][0])
                if original_delta["rows"][0]["actual_argmax"] != failure["first_divergence"]["left_token"] or original_delta["rows"][0]["reference_argmax"] != failure["first_divergence"]["right_token"]:
                    raise AssertionError("original mismatch was not reproduced exactly")
                prefix = snapshot["requests"][chosen]["kv"]
                effective = torch.stack([torch.stack((
                    torch.cat((prefix[0, layer], pt.records[f"layer_{layer:02d}.k_rope"][:offset]), 0),
                    torch.cat((prefix[1, layer], pt.records[f"layer_{layer:02d}.v_scattered"][:offset]), 0)))
                    for layer in range(prefix.shape[1])], dim=1)
                ar_prefix = ar_snapshot["requests"][ar_chosen]["kv"]
                effective_tokens = torch.cat((snapshot["requests"][chosen]["committed"][0, :-1],
                    snapshot["trees"][chosen]["token_ids"][path]), 0)
                if not torch.equal(effective_tokens, ar_snapshot["requests"][ar_chosen]["committed"][0]):
                    raise AssertionError("original disputed rows have different semantic tokens/positions")
                cache_comparison = [tensor_metrics(effective[:, layer], ar_prefix[:, layer])
                                    for layer in range(prefix.shape[1])]
                # Search chronologically, BEFORE the token mismatch, starting
                # from identical state for each round. Stop at first numeric break.
                origin = None
                for candidate_round, candidate in enumerate(packed_snapshots[:round_id + 1]):
                    candidate_chosen = next((i for i, r in enumerate(candidate["requests"]) if r["prompt_id"] == prompt), None)
                    if candidate_chosen is None:
                        continue
                    semantic_checks(candidate)
                    candidate_request = candidate["requests"][candidate_chosen]
                    candidate_path = [n for n in candidate["paths"][prompt]
                        if len(candidate_request["output_ids"]) + int(candidate["trees"][candidate_chosen]["depth"][n]) <= output_index]
                    base = packed_trace(runner, candidate, candidate_chosen, candidate_path)
                    serial, _ = serial_from_same_state(runner, candidate, candidate_chosen, candidate_path)
                    comparison = compare_path(base, serial)
                    first = comparison["earliest_chronological_node"]
                    if first is not None:
                        witness_path = candidate_path[:first["path_offset"] + 1]
                        layer = int(first["stage"].split('.')[0].split('_')[-1])
                        # Earlier nodes can have invisible-to-BF16 FP32 drift.
                        # Capture all layers so that their absence is measured,
                        # not inferred from equal materialized BF16 tensors.
                        captured_layers = tuple(range(len(runner.target.model.layers)))
                        witness = packed_trace(runner, candidate, candidate_chosen, witness_path, full_layers=captured_layers)
                        serial_witnesses, serial_state = serial_from_same_state(
                            runner, candidate, candidate_chosen, witness_path, full_layers=captured_layers,
                            capture_prefix_states=True)
                        operators = [attention_origin(runner, candidate, candidate_chosen, witness_path,
                            witness, serial_witnesses[-1], serial_state, selected_layer) for selected_layer in range(layer + 1)]
                        prior_controls = []
                        for path_offset in range(len(witness_path) - 1):
                            previous_path = witness_path[:path_offset + 1]
                            prior_controls.append({"path_offset": path_offset, "node": previous_path[-1],
                                "attention_controls_by_layer": [attention_origin(runner, candidate, candidate_chosen,
                                    previous_path, row_trace(witness, path_offset), serial_witnesses[path_offset],
                                    serial_state["earlier_states"][path_offset], selected_layer)
                                    for selected_layer in captured_layers]})
                        chronological_controls = prior_controls + [{"path_offset": len(witness_path) - 1,
                            "node": witness_path[-1], "attention_controls_by_layer": operators}]
                        first_fp32 = next(({"path_offset": item["path_offset"], "node": item["node"],
                            "layer": selected_layer, "stage": f"layer_{selected_layer:02d}.attention_output",
                            "max_abs": op["sparse_vs_compact_fp32"]["max_abs"]}
                            for item in chronological_controls
                            for selected_layer, op in enumerate(item["attention_controls_by_layer"])
                            if not op["sparse_vs_compact_fp32"]["bitwise_equal"]), None)
                        operator = operators[-1] if first["stage"].endswith("attention_output") else None
                        isolation = isolation_control(runner, candidate, candidate_chosen, witness_path, witness)
                        origin = {"round": candidate_round, "prefix_length": candidate["requests"][candidate_chosen]["cache_len"],
                            "total_q": candidate["query_offsets"][-1], "path": witness_path,
                            "prediction_output_index": len(candidate_request["output_ids"]) + len(witness_path) - 1,
                            "earliest_divergence": first, "layerwise_comparison": comparison,
                            "attention_operator_control": operator, "attention_controls_by_layer": operators,
                            "prior_node_all_layer_attention_controls": prior_controls,
                            "earliest_chronological_pre_round_fp32_attention_difference": first_fp32,
                            "earliest_pre_round_fp32_attention_difference_layer": next((i for i, op in enumerate(operators)
                                if not op["sparse_vs_compact_fp32"]["bitwise_equal"]), None),
                            "fixed_layout_isolation": isolation}
                        break
                if origin is None:
                    raise AssertionError("no identical-state origin found despite reproduced closed-loop mismatch")
                common_serial, _ = serial_from_same_state(runner, snapshot, chosen, path)
                common_compare = compare_path(pt, common_serial)
                entry = {"case": case, "prompt_id": prompt, "output_index": output_index,
                    "original_packed_round": round_id, "original_packed_path": path,
                    "current_full_token_prefix_identical_until_disputed_output": True,
                    "current_disputed_semantic_tokens_and_rope_positions_equal": True,
                    "original_closed_loop_layerwise": original_compare, "original_closed_loop_logit_witness": original_delta,
                    "original_packed_logit_top": logit_margin(pt.records["lm_head"][-1]),
                    "original_ar_logit_top": logit_margin(at.records["lm_head"][0]),
                    "original_effective_cache_by_layer": cache_comparison,
                    "original_first_cache_difference_layer": next((i for i, m in enumerate(cache_comparison) if not m["bitwise_equal"]), None),
                    "identical_state_causal_origin": origin,
                    "identical_state_at_mismatch_round": common_compare,
                    "identical_state_mismatch_round_logit_witness": argmax_flip_witness(
                        pt.records["lm_head"][-1], common_serial[-1].records["lm_head"][0]),
                    "original_shape_only_q_projection_control": shape_linear_control(runner, pt, at, offset)
                        if original_compare["earliest_divergence"]["stage"] == "layer_00.q_raw" else None}
                report["cases"].append(entry)
                save_json(args.output, report)
                print(case, prompt, "origin", origin["round"], first["stage"], "node", witness_path[-1],
                      "first_fp32_layer", origin["earliest_pre_round_fp32_attention_difference_layer"], flush=True)
            # Restore the capture wrappers for next generation pair.
            capture = Capture(runner)
    finally:
        capture.restore()
        runner.close()
        report["allocator_clean_after_close"] = not runner.block_manager.used_block_ids
        save_json(args.output, report)
    ending = identity(args)
    report["production_source_unchanged"] = source["production_source_sha256"] == ending["production_source_sha256"]
    report["diagnostic_file_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in diagnostic_names}
    report["diagnostic_source_unchanged"] = diagnostic_start == report["diagnostic_file_sha256"]
    all_attention_controls = [op for c in report["cases"]
        for item in c["identical_state_causal_origin"]["prior_node_all_layer_attention_controls"] +
                    [{"attention_controls_by_layer": c["identical_state_causal_origin"]["attention_controls_by_layer"]}]
        for op in item["attention_controls_by_layer"]]
    report["summary"] = {"six_original_mismatches_reproduced": len(report["cases"]) == 6,
        "all_independent_visibility_depth_checks_passed": True,
        "all_fixed_layout_layerwise_isolation_passed": all(all(c["identical_state_causal_origin"]["fixed_layout_isolation"].values()) for c in report["cases"]),
        "all_attention_pre_round_oracle_bounds_passed": all(c["identical_state_causal_origin"]["attention_operator_control"] is not None and
            c["identical_state_causal_origin"]["attention_operator_control"]["sparse_vs_fp64"]["within_fixed_bound"] and
            c["identical_state_causal_origin"]["attention_operator_control"]["compact_vs_fp64"]["within_fixed_bound"] for c in report["cases"]),
        "allocator_clean_after_close": report["allocator_clean_after_close"],
        "production_source_unchanged": report["production_source_unchanged"],
        "diagnostic_source_unchanged": report["diagnostic_source_unchanged"],
        "all_pre_round_layer_oracle_bounds_passed": all(op["sparse_vs_fp64"]["within_fixed_bound"] and
            op["compact_vs_fp64"]["within_fixed_bound"] for op in all_attention_controls),
        "all_actual_serial_roundtrips_exact": all(op["actual_serial_bf16_equals_compact_fp32_rounded"]
            for op in all_attention_controls),
        "all_packed_roundtrips_exact": all(op["original_bf16_equals_fp32_replay_rounded"]
            for op in all_attention_controls),
        "attention_operator_controls_checked": len(all_attention_controls),
        "same_round_start_kv_removes_all_six_disputed_argmax_flips": all(
            c["identical_state_mismatch_round_logit_witness"]["argmax_flips"] == 0 for c in report["cases"])}
    save_json(args.output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--draft", default=DRAFT)
    parser.add_argument("--oracle", default=ORACLE)
    parser.add_argument("--qualification", default=str(Path(__file__).parent / "phase31_qualification.json"))
    parser.add_argument("--original", default="/root/autodl-tmp/benchmarks/jetspec-phase31/final-stable.json")
    parser.add_argument("--output", default="/root/autodl-tmp/benchmarks/jetspec-phase31-numerics/diagnostic.json")
    run(parser.parse_args())
