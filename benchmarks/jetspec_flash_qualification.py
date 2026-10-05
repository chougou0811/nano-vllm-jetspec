#!/usr/bin/env python3
"""Untimed trained-Qwen3 gates for opt-in Flash Draft attention.

This is not a benchmark. The predeclared BF16 envelope is the frozen Phase-4
2^-6 scaled-max/RMS gate, never an adaptive threshold. Production prefill stays
SDPA; the rejected whole-network FA prefill experiment is recorded separately.
An isolated trained layer-0 FA control is NOT full-network prefill qualification.
Tree attention keeps the existing FP32/TILE64 implementation and exact gates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import jetspec_phase4_qualification as frozen
import jetspec_phase32 as shared
import jetspec_chunked_prefill as chunks
from jetspec_phase3 import save_json

BF16_BOUND = 2 ** -6
TREE_REVISION = "b388330"


def metrics(actual, reference):
    return frozen.tensor_metrics(actual, reference, bound=BF16_BOUND)


def configure(engine, args, concurrency=8, *, backend="flash_attn", chunk=0):
    frozen.require(backend in ("sdpa", "flash_attn"), "unknown qualification backend")
    frozen.require(engine.is_finished(), "qualification configuration requires idle engine")
    engine.configure_jetspec(args.draft, optimization="serving", attention_backend=backend,
        max_admissions_per_step=8, enable_chunked_prefill=bool(chunk),
        prefill_chunk_size=chunk or 256, max_prefill_tokens=chunk or args.max_model_len)
    engine._jetspec_scheduler.max_num_seqs = concurrency
    runtime = engine._jetspec_scheduler.runtime
    frozen.require(runtime._attention_backend == backend, "backend selection was reset")
    frozen.require(runtime._prefill_attention_backend == "sdpa", "unqualified Flash prefill entered serving")
    return runtime


@contextmanager
def flash_helpers(engine, args):
    """Preserve explicit policy when old untimed helpers reconfigure serving."""
    def mode(selected_engine, name, draft, concurrency):
        frozen.require(selected_engine is engine and name == "jetspec" and draft == args.draft,
                       "unexpected qualification mode switch")
        configure(engine, args, concurrency)

    def chunk_mode(selected_engine, selected_args, concurrency, chunk):
        frozen.require(selected_engine is engine, "unexpected qualification engine")
        return configure(engine, selected_args, concurrency, chunk=chunk)

    with frozen.patch(shared, "set_mode", mode), frozen.patch(chunks, "configure", chunk_mode):
        yield


def provenance(args):
    result = frozen.qualification_identity(args)
    result["phase4_qualification_sha256"] = result["script_sha256"]
    script = Path(__file__).resolve()
    helpers = {Path(module.__file__).resolve() for module in (frozen, shared, chunks)}
    result["flash_qualification_script_path"] = str(script)
    result["script_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
    result["flash_qualification_helper_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(helpers)}
    result["flash_harness_sha256"] = hashlib.sha256(json.dumps({
        "script": result["script_sha256"], "direct_helpers": result["flash_qualification_helper_sha256"],
        "inherited_helpers": result["qualification_helper_sha256"]}, sort_keys=True).encode()).hexdigest()
    relative = "nanovllm/speculative/jetspec/paged_backend.py"
    actual = Path(args.repo).resolve() / relative
    baseline = subprocess.check_output(["git", "-C", args.repo, "show", f"{TREE_REVISION}:{relative}"])
    result["tree_backend"] = {"path": str(actual), "sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
        "frozen_revision": TREE_REVISION, "frozen_sha256": hashlib.sha256(baseline).hexdigest(),
        "unchanged_from_frozen": actual.read_bytes() == baseline,
        "contract": "FP32 multiply/reduction/online-softmax TILE64; pre-BF16 FP64 oracle gates unchanged"}
    frozen.require(result["tree_backend"]["unchanged_from_frozen"], "qualified tree backend was changed")
    return result


def release(runtime, requests):
    for request in requests:
        if runtime.requests.get(request.request_id) is request:
            runtime.cancel(request)
    for context in list(runtime.prefills.values()):
        runtime.cancel_prefill(context)
    runtime.release_idle_scratch()


def prefix_snapshot(runtime, ids, plan, request_id):
    """Capture actual paged all-layer KV/taps and lm_head at a fixed chunk plan."""
    captured = []
    hook = runtime.target.lm_head.register_forward_hook(
        lambda _m, _a, out: captured.append(out.detach().cpu().reshape(-1, out.shape[-1])[-1].clone()))
    request, context, history_checks = None, None, []
    try:
        if plan is None:
            request = runtime.create_request(ids, max_new_tokens=32, tree_budget=31,
                                            ignore_eos=True, request_id=request_id)
            records = [{"backend": runtime._prefill_attention_backend, "tokens": len(ids)}]
        else:
            context = runtime.begin_prefill(ids, max_new_tokens=32, tree_budget=31,
                                           ignore_eos=True, request_id=request_id)
            for count in plan:
                slots = context.logical_slots
                old = chunks.raw_kv(runtime, slots) if slots.numel() else None
                request = runtime.prefill_step(context, count)
                if old is not None:
                    exact = chunks.bitwise_equal(old, chunks.raw_kv(runtime, slots))
                    history_checks.append(exact)
                    frozen.require(exact, "Flash/SDPA chunk overwrote committed historical KV")
            frozen.require(request is not None and context.promoted, "chunk plan did not promote")
            records = list(context.chunk_records)
            expected = "paged_layerwise_dense_chunk"
            frozen.require(all(row["backend"] == expected for row in records), "selected chunk backend not executed")
        request.state.assert_round_invariant(validate_device=True)
        frozen.require(captured, "prefill lm_head was not observed")
        # SDPA diagnostic applied to both feature sets. Anchors are aligned later
        # so a near-tie Target argmax cannot change the Draft comparator input.
        return {"kv": chunks.raw_kv(runtime, request.state.logical_slots),
                "taps": request.state.target_hidden.detach().cpu().clone(),
                "logits": captured[-1], "tokens": request.state.committed.detach().cpu().clone(),
                "chunk_records": records, "history_checks": history_checks}
    finally:
        hook.remove()
        release(runtime, [] if request is None else [request])


def prefill_qualification(engine, args, report):
    """Production prefill must remain bitwise unchanged under Draft dispatch."""
    cases = [(33, None), (257, None), (1024, None), (2048, None),
             (33, [16, 1, 16]), (257, [255, 1, 1]), (515, [256, 256, 3])]
    for index, (length, plan) in enumerate(cases):
        runtime = configure(engine, args, backend="sdpa")
        ids = chunks.prompt(runtime.tokenizer, length, index)
        reference = prefix_snapshot(runtime, ids, plan, f"sdpa-prefix-{index}")
        runtime = configure(engine, args)
        actual = prefix_snapshot(runtime, ids, plan, f"flash-prefix-{index}")
        repeated = prefix_snapshot(runtime, ids, plan, f"flash-prefix-repeat-{index}")
        kv = [metrics(actual["kv"][:, layer], reference["kv"][:, layer])
              for layer in range(actual["kv"].shape[1])]
        check = {"prompt_length": length, "chunk_plan": plan, "kv_by_layer": kv,
                 "taps": metrics(actual["taps"], reference["taps"]),
                 "target_logits": metrics(actual["logits"], reference["logits"]),
                 "target_argmax": chunks.argmax_witness(actual["logits"], reference["logits"]),
                 "same_shape_repeat_bitwise_exact": {key: chunks.bitwise_equal(actual[key], repeated[key])
                    for key in ("kv", "taps", "logits", "tokens")},
                 "history_byte_exact": all(actual["history_checks"]), "chunk_records": actual["chunk_records"],
                 "sdpa_prefill_vs_flash_draft_mode_bitwise_exact": {key: chunks.bitwise_equal(actual[key], reference[key])
                    for key in ("kv", "taps", "logits", "tokens")},
                 "executed_prefill_backend": "sdpa", "qualification_scope": "SDPA prefill unchanged; not FA prefill"}
        report["prefill"].append(check)
        save_json(args.output, report)
        frozen.require(all(check["sdpa_prefill_vs_flash_draft_mode_bitwise_exact"].values()), "Flash Draft mode changed production SDPA prefill")
        frozen.require(all(check["same_shape_repeat_bitwise_exact"].values()), "same-shape Flash prefill replay changed")
        print(f"SDPA prefill P{length} plan={plan}: Draft-mode bitwise/repeat/history PASS", flush=True)
        del actual, reference, repeated


def rejected_prefill_evidence(path, output):
    path = Path(path).resolve()
    frozen.require(path != Path(output).resolve(), "must not overwrite rejected prefill evidence")
    report = json.loads(path.read_text())
    frozen.require(report.get("passed") is False and report.get("status") == "failed",
                   "required FA prefill negative result is not a failed experiment")
    frozen.require(report.get("bf16_scaled_max_and_relative_rms_bound") == BF16_BOUND,
                   "rejected experiment used a different numerical gate")
    failed = [row for row in report.get("prefill", []) if any(not metric["passed"] for metric in
        [*row["kv_by_layer"], row["taps"], row["target_logits"], row["same_anchor_feature_draft_logits"]])]
    frozen.require(failed and report.get("error") and report.get("source"), "negative experiment lacks fixed-gate failing witnesses")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "status": "rejected", "qualified": False, "used_in_serving": False,
            "fixed_bf16_bound": BF16_BOUND, "source": report["source"],
            "error": report["error"], "failed_prefill_cases": failed,
            "environment": report["environment"],
            "note": "Whole-network FA prefill failed the predeclared envelope. Local layer-0 controls cannot overturn this rejection."}


def causal_fp64(q, k, v, scale):
    """Independent all-row CPU FP64 GQA attention, no SDPA or production mask."""
    import torch
    q, k, v = [tensor.detach().cpu().double() for tensor in (q, k, v)]
    groups = q.shape[1] // k.shape[1]
    frozen.require(q.shape[1] % k.shape[1] == 0 and q.shape[0] == k.shape[0], "isolated control expects square GQA")
    keys = k.repeat_interleave(groups, dim=1).transpose(0, 1)
    values = v.repeat_interleave(groups, dim=1).transpose(0, 1)
    scores = q.transpose(0, 1) @ keys.transpose(1, 2) * scale
    allowed = torch.arange(q.shape[0])[:, None] >= torch.arange(k.shape[0])[None, :]
    weights = scores.masked_fill(~allowed[None], float("-inf")).softmax(-1)
    return (weights @ values).transpose(0, 1)


def trained_prefill_kernel_control(runtime):
    """Test mask/operator semantics on real layer-0 operands, not full prefill."""
    import torch
    import torch.nn.functional as F
    from nanovllm.speculative.jetspec import flash_prefill
    captured = []
    original = flash_prefill.flash_causal_prefill

    def observe(q, k, v, metadata, scale):
        output = original(q, k, v, metadata, scale)
        if not captured:
            captured.append((q.detach().clone(), k.detach().clone(), v.detach().clone(),
                             output.detach().clone(), metadata, scale))
        return output

    ids = torch.tensor(chunks.prompt(runtime.tokenizer, 33), device=runtime.kv_pool.device)
    with torch.inference_mode(), frozen.patch(flash_prefill, "flash_causal_prefill", observe):
        runtime.target.model.forward_dense(ids, torch.arange(ids.numel(), device=ids.device),
            None, None, runtime.target_layer_ids, attention_backend="flash_attn")
    frozen.require(len(captured) == 1, "trained layer-0 Flash operands were not observed")
    q, k, v, actual, metadata, scale = captured[0]
    with torch.inference_mode():
        repeated = original(q, k, v, metadata, scale)
        sdpa = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0, is_causal=True, scale=scale, enable_gqa=q.shape[1] != k.shape[1])
        sdpa = sdpa.squeeze(0).transpose(0, 1).contiguous()
        changed_k, changed_v = k.clone(), v.clone()
        boundary = 16
        changed_k[boundary:].add_(7)
        changed_v[boundary:].sub_(3)
        changed = original(q, changed_k, changed_v, metadata, scale)
    oracle = causal_fp64(q, k, v, scale)
    first_value = v[0].repeat_interleave(q.shape[1] // v.shape[1], dim=0)
    check = {"scope": "trained layer 0, identical post-RoPE Q/K/V; NOT whole-network FA prefill qualification",
        "query_shape": list(q.shape), "key_shape": list(k.shape), "scale": scale,
        "fa_vs_cpu_fp64": metrics(actual, oracle), "sdpa_vs_cpu_fp64": metrics(sdpa, oracle),
        "fa_vs_sdpa": metrics(actual, sdpa), "same_shape_repeat_bitwise_exact": chunks.bitwise_equal(actual, repeated),
        "row0_equals_only_visible_v0_bitwise": chunks.bitwise_equal(actual[0], first_value),
        "future_kv_perturbation_prefix_bitwise_exact": chunks.bitwise_equal(actual[:boundary], changed[:boundary]),
        "future_kv_perturbation_tail_changed": not chunks.bitwise_equal(actual[boundary:], changed[boundary:]),
        "local_fixed_bf16_bound": BF16_BOUND, "full_network_prefill_qualified": False}
    frozen.require(all(check[name]["passed"] for name in ("fa_vs_cpu_fp64", "sdpa_vs_cpu_fp64", "fa_vs_sdpa")),
                   f"local same-operand Flash prefill kernel failed fixed gate: {check}")
    frozen.require(all(check[name] for name in ("same_shape_repeat_bitwise_exact", "row0_equals_only_visible_v0_bitwise",
        "future_kv_perturbation_prefix_bitwise_exact", "future_kv_perturbation_tail_changed")), f"local causal mask/precision control failed: {check}")
    return check


def compare_draft(runtime, requests, label):
    import torch
    from nanovllm.speculative.jetspec.batched_draft import BatchedDraftProposer
    from nanovllm.speculative.jetspec.flash_draft import FlashDraftProposer
    from jetspec_numeric_oracles import argmax_flip_witness
    grouped = [frozen.clone_request(runtime, request) for request in requests]
    serial = [frozen.clone_request(runtime, request) for request in requests]
    old = [[(k.clone(), v.clone()) for k, v in request.drafter._fwd.cache] for request in requests]
    reference = BatchedDraftProposer(runtime.head, runtime.target).propose(grouped, runtime.tree_depth)
    serial_logits = [frozen.serial(request, runtime.tree_depth) for request in serial]
    proposer = FlashDraftProposer(runtime.head, runtime.target)
    actual = proposer.propose(requests, runtime.tree_depth)
    rows, owners = [], set()
    for i, (request, control, got, ref, scalar, prior) in enumerate(zip(requests, grouped, actual, reference, serial_logits, old)):
        check = {"index": i, "context_length": int(request.state.target_hidden.shape[1]),
            "logits_vs_grouped_sdpa": metrics(got, ref), "logits_vs_serial_sdpa": metrics(got, scalar),
            "argmax_vs_grouped_sdpa": argmax_flip_witness(got.cpu().reshape(-1, got.shape[-1]), ref.cpu().reshape(-1, ref.shape[-1])),
            "kv": [], "old_cache_byte_exact": True, "compact_cache": True,
            "cache_cropped_to_context": request.drafter._fwd.cache.get_seq_length() == request.state.target_hidden.shape[1]}
        for layer, (pair, ref_pair) in enumerate(zip(request.drafter._fwd.cache, control.drafter._fwd.cache)):
            check["kv"].append({"layer": layer, "key": metrics(pair[0], ref_pair[0]), "value": metrics(pair[1], ref_pair[1])})
            for part, tensor in enumerate(pair):
                address = tensor.untyped_storage().data_ptr()
                frozen.require(address not in owners, "Flash caches alias across requests/layers")
                owners.add(address)
                check["compact_cache"] &= tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
                if prior:
                    check["old_cache_byte_exact"] &= chunks.bitwise_equal(tensor[:, :, :prior[layer][part].shape[-2]], prior[layer][part])
        rows.append(check)
    result = {"label": label, "stats": proposer.last_stats, "checks": rows}
    frozen.require(proposer.last_stats["serial_forward_calls"] == 0 and
                   proposer.last_stats["attention_key_padding_slots"] == 0, "Flash Draft fell back or retained padded attention keys")
    for check in rows:
        frozen.require(check["logits_vs_grouped_sdpa"]["passed"] and check["logits_vs_serial_sdpa"]["passed"] and
            all(row[name]["passed"] for row in check["kv"] for name in ("key", "value")), f"fixed 2^-6 Draft envelope failed: {result}")
        frozen.require(check["old_cache_byte_exact"] and check["compact_cache"] and check["cache_cropped_to_context"],
                       f"Flash Draft cache ownership failed: {result}")
    return result


def draft_isolation(runtime, requests):
    import torch
    from nanovllm.speculative.jetspec.flash_draft import FlashDraftProposer
    base = [frozen.clone_request(runtime, request) for request in requests]
    changed = [frozen.clone_request(runtime, request) for request in requests]
    for request in changed[1:]:
        request.state.target_hidden.add_(7)
        request.state.committed[:, -1].add_(1).remainder_(runtime.target.lm_head.weight.shape[0])
        for keys, values in request.drafter._fwd.cache:
            keys.add_(2)
            values.sub_(3)
    proposer = FlashDraftProposer(runtime.head, runtime.target)
    expected = proposer.propose(base, runtime.tree_depth)
    before = dict(proposer.last_stats)
    actual = proposer.propose(changed, runtime.tree_depth)
    same = torch.equal(expected[0], actual[0])
    caches = all(chunks.bitwise_equal(a, b) for x, y in zip(base[0].drafter._fwd.cache, changed[0].drafter._fwd.cache) for a, b in zip(x, y))
    frozen.require(before["batch_sizes"] == proposer.last_stats["batch_sizes"] and any(n > 1 for n in before["batch_sizes"]),
                   "Draft isolation control did not preserve a real multi-request group")
    frozen.require(same and caches, "finite other-request mutation contaminated Flash Draft")
    return {"chosen_logits_bitwise_exact": same, "chosen_cache_bitwise_exact": caches,
            "batch_sizes": before["batch_sizes"], "attention_key_padding_slots": before["attention_key_padding_slots"]}


def draft_qualification(runtime, args, report):
    budgets = (63, 31, 47, 63)
    for lengths, label in (((35,), "singleton"), ((27, 43, 35, 39), "c4-small"),
                           ((1020, 1024, 1028, 1022), "c4-near1k"), ((27, 1024, 43, 2048), "c4-long-ragged")):
        requests = [runtime.create_request(chunks.prompt(runtime.tokenizer, length, i),
            max_new_tokens=96, tree_budget=budgets[i], ignore_eos=True, request_id=f"{label}:{i}")
            for i, length in enumerate(lengths)]
        try:
            report["draft"].append(compare_draft(runtime, requests, label + "-cold"))
            runtime.step(requests)  # Real accepted-only commit supplies the next new-context suffix.
            for request in requests:
                request.state.assert_round_invariant(validate_device=True)
            report["draft"].append(compare_draft(runtime, requests, label + "-accepted-suffix"))
            for request, delta in zip(requests, (1, 2, 3, 2)):
                request.drafter._fwd.cache.crop(request.state.cache_len - delta)
            report["draft"].append(compare_draft(runtime, requests, label + "-warm-cropped"))
            if label == "c4-near1k":
                report["draft_isolation"] = draft_isolation(runtime, requests)
            save_json(args.output, report)
            print(f"Flash Draft {label}: cold/accepted-suffix/warm cache gates PASS", flush=True)
        finally:
            release(runtime, requests)


def lifecycle_semantics(result, eos_ids):
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


def run(args):
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    from nanovllm import LLM
    source = provenance(args)
    rejected = rejected_prefill_evidence(args.rejected_prefill_results, args.output)
    engine = None
    report = {"kind": "untimed trained-model Flash Draft qualification; production prefill remains SDPA", "source": source,
              "bf16_scaled_max_and_relative_rms_bound": BF16_BOUND, "tree_contract_unchanged": True,
              "cross_backend_token_bitwise_required": False, "prefill": [], "draft": [],
              "all_gates_passed": False, "passed": False, "status": "running",
              "rejected_whole_network_flash_prefill": rejected, "production_prefill_backend": "sdpa"}
    save_json(args.output, report)
    try:
        torch.manual_seed(0)
        engine = LLM(args.target, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=.8,
            max_num_seqs=8, max_model_len=args.max_model_len, max_num_batched_tokens=args.max_model_len,
            kvcache_block_size=256)
        import nanovllm, flash_attn
        frozen.require(Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()), "wrong selected production import")
        report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "flash_attn": flash_attn.__version__, "dtype": "bfloat16"}
        prefill_qualification(engine, args, report)
        runtime = configure(engine, args)
        frozen.require(all(rejected["environment"][key] == report["environment"][key] for key in
            ("torch", "cuda", "gpu", "flash_attn", "dtype")), "negative result execution environment differs")
        report["trained_layer0_flash_prefill_kernel_control"] = trained_prefill_kernel_control(runtime)
        save_json(args.output, report)
        report["partial_prefill_isolation"] = chunks.isolation_qualification(runtime)
        draft_qualification(runtime, args, report)
        prompts = {row["id"]: row for row in json.loads(Path(args.oracle).read_text())["prompts"]}
        with flash_helpers(engine, args):
            report["packed_tree_isolation_commit"] = shared.eight_request_isolation(engine, prompts, args.draft)
            isolation = report["packed_tree_isolation_commit"]
            frozen.require(isolation["real_eight_request_packed_shape"] and isolation["finite_same_shape_controls_passed"], "Flash pipeline tree isolation failed")
            frozen.require(all(isolation["checks"]["all_layer_raw_copy_checks"]) and all(isolation["checks"]["history_and_rejected_checks"]), "Flash pipeline raw commit/history failed")
            configure(engine, args, 2)
            with shared.ServingProbe(engine, diagnostic=True) as probe:
                first = shared.serving_run(engine, shared.qualification_workload(prompts), mode="jetspec", clock="step", label="flash-lifecycle-a", max_wall_s=args.deadline)
                checks = probe.metrics()
            second = shared.serving_run(engine, shared.qualification_workload(prompts), mode="jetspec", clock="step", label="flash-lifecycle-b", max_wall_s=args.deadline)
            exact = shared.replay_signature(first) == shared.replay_signature(second)
            report["lifecycle"] = {"first": first, "second": second, "checks": checks, "same_schedule_replay_exact": exact}
            report["lifecycle"]["semantic_gates"] = lifecycle_semantics(first, runtime.eos_token_ids)
            frozen.require(exact and first["dynamic_live_arrival_seen"] and all(checks["all_layer_raw_copy_checks"]) and all(checks["history_and_rejected_checks"]), "Flash lifecycle replay/dynamic/copy gates failed")
            report["allocator_pressure"] = shared.allocator_pressure(engine, prompts)
            frozen.require(all(report["allocator_pressure"][key] for key in ("deferred_seen", "recovered")), "held-page admission did not recover")
            report["preemption_recompute"] = shared.preemption_qualification(engine, prompts)
            frozen.require(all(report["preemption_recompute"][key] for key in ("preemption_seen", "resume_seen", "output_exactly_once", "fixed_schedule_replay_exact", "scratch_one_page_or_less")), "Flash recompute/preemption gate failed")
            report["chunked_lifecycle_recompute"] = chunks.lifecycle_qualification(engine, args)
        runtime = engine._jetspec_scheduler.runtime
        frozen.require(runtime._attention_backend == "flash_attn", "qualification silently reset Flash policy")
        frozen.require(engine.is_finished() and not runtime.requests and not runtime.prefills, "qualification retained live requests")
        engine.disable_jetspec()
        report["allocator_clean"] = not engine.scheduler.block_manager.used_block_ids
        frozen.require(report["allocator_clean"], "Flash qualification leaked canonical/scratch pages")
        after = provenance(args)
        report["source_unchanged"] = after["production_source_sha256"] == source["production_source_sha256"]
        report["harness_unchanged"] = after["flash_harness_sha256"] == source["flash_harness_sha256"]
        frozen.require(report["source_unchanged"] and report["harness_unchanged"], "sources changed during qualification")
        report["all_gates_passed"] = True
        report["passed"] = True
        report["status"] = "complete"
        save_json(args.output, report)
        print("ALL untimed trained-model Flash gates PASS", flush=True)
    except BaseException as error:
        report["passed"] = False
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save_json(args.output, report)
        raise
    finally:
        if engine is not None:
            scheduler = getattr(engine, "_jetspec_scheduler", None)
            if scheduler is not None:
                for request_id in list(scheduler.requests):
                    engine.cancel_request(request_id)
                release(scheduler.runtime, list(scheduler.runtime.requests.values()))
                scheduler.drain_events()
            engine.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--target", default=shared.TARGET)
    parser.add_argument("--draft", default=shared.DRAFT)
    parser.add_argument("--oracle", default=shared.ORACLE)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--deadline", type=float, default=600)
    parser.add_argument("--rejected-prefill-results", required=True,
                        help="Preserved failed whole-network FA prefill qualification JSON; never overwritten")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
