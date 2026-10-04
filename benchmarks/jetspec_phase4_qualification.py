#!/usr/bin/env python3
"""Untimed trained-model gates for Phase-4 Draft/metadata/feature changes.

This diagnostic deliberately copies/poisons tensors and forces device checks.
It is not a performance benchmark and makes no cross-shape token identity claim.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from jetspec_phase3 import save_json
from jetspec_phase31 import identity
import jetspec_phase32 as shared


def qualification_identity(args):
    """Pin the selected production tree and the actually loaded diagnostic code.

    --repo can select a clean checkout while this script/helpers live outside
    it. Their explicit paths/hashes must not be attributed to that checkout's
    Git revision. External checkpoint bytes and CUDA library binaries are not
    covered by this source fingerprint.
    """
    import jetspec_phase3
    import jetspec_phase31
    import jetspec_numeric_oracles
    result = identity(args)
    result["phase31_identity_script_sha256"] = result["script_sha256"]
    script = Path(__file__).resolve()
    result["qualification_script_path"] = str(script)
    result["script_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
    helpers = {Path(module.__file__).resolve() for module in (
        jetspec_phase3, jetspec_phase31, shared, jetspec_numeric_oracles)}
    result["qualification_helper_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(helpers)}
    result["qualification_harness_source_sha256"] = hashlib.sha256(json.dumps({
        "script": result["script_sha256"], "helpers": result["qualification_helper_sha256"]
    }, sort_keys=True).encode()).hexdigest()
    result["qualification_harness_external_to_selected_checkout"] = not script.is_relative_to(Path(args.repo).resolve())
    result["oracle_sha256"] = hashlib.sha256(Path(args.oracle).read_bytes()).hexdigest()
    result["fingerprint_scope"] = {
        "covered": "selected production Python, loaded official JetSpec Python, this script and all direct/transitive benchmark helpers, oracle input",
        "not_covered": "external Target/Draft checkpoint weight bytes, CUDA/cuBLAS/shared-library binaries",
        "model_paths": {"target": args.target, "draft": args.draft},
        "revision_scope": "revision belongs to --repo production tree; external diagnostic paths have separate explicit hashes"}
    return result


@contextmanager
def patch(obj, name, value):
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def tensor_metrics(actual, reference, *, bound=2 ** -6):
    import torch
    a, b = actual.detach().cpu().double(), reference.detach().cpu().double()
    require(a.shape == b.shape, "Draft comparator shapes differ")
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "nonfinite Draft operands")
    delta = a - b
    absolute = float(delta.abs().max()) if a.numel() else 0.0
    rms = float(delta.square().mean().sqrt()) if a.numel() else 0.0
    reference_rms = float(b.square().mean().sqrt()) if b.numel() else 0.0
    reference_max = float(b.abs().max()) if b.numel() else 0.0
    relative = rms / reference_rms if reference_rms else (0.0 if not rms else float("inf"))
    passed = absolute <= bound * max(1, reference_max) and relative <= bound
    return {"max_abs_error": absolute, "rms_error": rms, "relative_rms_error": relative,
            "reference_absmax": reference_max, "reference_rms": reference_rms,
            "fixed_scaled_max_and_relative_rms_bound": bound, "passed": passed}


def clone_request(runtime, request):
    from transformers import DynamicCache
    result = SimpleNamespace(state=SimpleNamespace(
        committed=request.state.committed.clone(), target_hidden=request.state.target_hidden.clone()),
        drafter=runtime._new_drafter())
    result.drafter._fwd.cache = DynamicCache.from_legacy_cache(tuple(
        (k.clone(), v.clone()) for k, v in request.drafter._fwd.cache))
    return result


def serial(request, depth):
    return request.drafter.propose_logits(request.state.committed, depth,
                                         target_hidden=request.state.target_hidden)


def compare_draft(runtime, requests, label):
    import torch
    from nanovllm.speculative.jetspec.batched_draft import BatchedDraftProposer
    from jetspec_numeric_oracles import argmax_flip_witness
    proposer = BatchedDraftProposer(runtime.head, runtime.target)
    refs = [clone_request(runtime, request) for request in requests]
    history = [[(k.clone(), v.clone()) for k, v in request.drafter._fwd.cache] for request in requests]
    expected = [serial(request, runtime.tree_depth) for request in refs]
    grouped = []
    original_batch = proposer._batch

    def observe(rows, depth):
        grouped.extend(row.index for row in rows)
        return original_batch(rows, depth)

    with patch(proposer, "_batch", observe):
        actual = proposer.propose(requests, runtime.tree_depth)
    checks = []
    for i, (got, ref, request, reference, old) in enumerate(zip(actual, expected, requests, refs, history)):
        metrics = tensor_metrics(got, ref)
        flips = argmax_flip_witness(got.cpu().reshape(-1, got.shape[-1]),
                                    ref.cpu().reshape(-1, ref.shape[-1]))
        kv = []
        compact = True
        unchanged = True
        for layer, ((keys, values), (ref_k, ref_v)) in enumerate(zip(
                request.drafter._fwd.cache, reference.drafter._fwd.cache)):
            km, vm = tensor_metrics(keys, ref_k), tensor_metrics(values, ref_v)
            kv.append({"layer": layer, "key": km, "value": vm})
            if i in grouped:
                compact &= all(t.untyped_storage().nbytes() == t.numel() * t.element_size()
                               for t in (keys, values))
            if old:
                old_k, old_v = old[layer]
                unchanged &= torch.equal(keys[:, :, :old_k.shape[-2]], old_k)
                unchanged &= torch.equal(values[:, :, :old_v.shape[-2]], old_v)
        cropped = request.drafter._fwd.cache.get_seq_length() == request.state.target_hidden.shape[1]
        check = {"request_index": i, "context_length": int(request.state.target_hidden.shape[1]),
                 "batched": i in grouped, "logits": metrics, "argmax": flips, "kv": kv,
                 "cache_cropped_to_real_context": cropped, "compact_storage": bool(compact),
                 "historical_prefix_byte_exact": bool(unchanged)}
        require(metrics["passed"] and all(c["key"]["passed"] and c["value"]["passed"] for c in kv),
                f"Draft BF16 envelope failed: {label} request{i} {check}")
        require(cropped and compact and unchanged, f"Draft cache ownership/history failed: {label} request{i}")
        checks.append(check)
    return {"label": label, "stats": proposer.last_stats, "checks": checks,
            "note": "BF16 cross-shape empirical envelope, not bitwise equivalence or a dot-product error theorem"}


def draft_isolation(runtime, requests):
    import torch
    from nanovllm.speculative.jetspec.batched_draft import BatchedDraftProposer, _RaggedCache
    base = [clone_request(runtime, request) for request in requests]
    changed = [clone_request(runtime, request) for request in requests]
    for request in changed[1:]:
        request.state.target_hidden.add_(7)
        request.state.committed[:, -1].add_(1).remainder_(runtime.target.lm_head.weight.shape[0])
        for keys, values in request.drafter._fwd.cache:
            keys.add_(2)
            values.sub_(3)
    proposer = BatchedDraftProposer(runtime.head, runtime.target)
    expected = proposer.propose(base, runtime.tree_depth)
    expected_stats = dict(proposer.last_stats)
    actual = proposer.propose(changed, runtime.tree_depth)
    require(expected_stats["batch_sizes"] == proposer.last_stats["batch_sizes"], "Draft perturbation changed groups")
    exact = torch.equal(expected[0], actual[0])
    cache_exact = all(torch.equal(a, b) for pair_a, pair_b in zip(
        base[0].drafter._fwd.cache, changed[0].drafter._fwd.cache) for a, b in zip(pair_a, pair_b))
    require(exact and cache_exact, "same-shape other-request Draft mutation contaminated chosen request")
    clean = [clone_request(runtime, request) for request in requests]
    poisoned = [clone_request(runtime, request) for request in requests]
    clean_logits = proposer.propose(clean, runtime.tree_depth)
    original_update = _RaggedCache.update
    pad_regions = []

    def poison_inputs(module, args, kwargs):
        cache = kwargs["past_key_values"]
        for i, row in enumerate(cache.rows):
            kwargs["target_hidden"][i, row.suffix_length:] = 1234
            kwargs["position_ids"][i, row.suffix_length:cache.suffix_capacity] = 237

    def poison_kv(cache, keys, values, layer, cache_kwargs=None):
        k, v = original_update(cache, keys, values, layer, cache_kwargs)
        for i, row in enumerate(cache.rows):
            regions = [(row.cached_length, cache.prefix_capacity),
                       (cache.prefix_capacity + row.suffix_length, cache.prefix_capacity + cache.suffix_capacity)]
            for begin, end in regions:
                if begin < end:
                    k[i, :, begin:end] = 13
                    v[i, :, begin:end] = -17
                    pad_regions.append([i, layer, begin, end])
        return k, v

    hook = runtime.head.register_forward_pre_hook(poison_inputs, with_kwargs=True)
    try:
        with patch(_RaggedCache, "update", poison_kv):
            poison_logits = proposer.propose(poisoned, runtime.tree_depth)
    finally:
        hook.remove()
    pad_exact = all(torch.equal(a, b) for a, b in zip(clean_logits, poison_logits))
    pad_cache_exact = all(torch.equal(a, b) for ra, rb in zip(clean, poisoned)
        for pair_a, pair_b in zip(ra.drafter._fwd.cache, rb.drafter._fwd.cache) for a, b in zip(pair_a, pair_b))
    require(pad_regions and pad_exact and pad_cache_exact, "masked old/new Draft padding affected output/cache")
    return {"same_shape_other_request_logits_bitwise_exact": exact,
            "same_shape_chosen_cache_bitwise_exact": cache_exact, "batch_sizes": expected_stats["batch_sizes"],
            "finite_old_and_new_padding_poison_regions": pad_regions,
            "poison_logits_bitwise_exact": pad_exact, "poison_cache_bitwise_exact": pad_cache_exact}


def new_requests(runtime, prompt, lengths, label):
    return [runtime.create_request((prompt * ((length + len(prompt) - 1) // len(prompt)))[:length],
        max_new_tokens=64, tree_budget=budget, ignore_eos=True, request_id=f"{label}:{i}")
        for i, (length, budget) in enumerate(zip(lengths, (63, 31, 47)))]


def release_requests(runtime, requests):
    for request in requests:
        if runtime.requests.get(request.request_id) is request:
            runtime.cancel(request)
    runtime.release_idle_scratch()


def run(args):
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    from nanovllm import LLM
    from nanovllm.speculative.jetspec import state as state_module
    from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata
    from jetspec_numeric_oracles import parent_chain
    source = qualification_identity(args)
    prompts = {item["id"]: item for item in json.loads(Path(args.oracle).read_text())["prompts"]}
    torch.manual_seed(0)
    engine = None
    report = {"phase": 4, "kind": "untimed trained-model qualification", **source,
              "bf16_draft_scaled_max_and_relative_rms_bound": 2 ** -6,
              "cross_shape_token_bitwise_required": False, "draft": [], "metadata_checks": []}
    save_json(args.output, report)
    try:
        engine = LLM(args.target, enforce_eager=True, tensor_parallel_size=1,
            gpu_memory_utilization=0.8, max_num_seqs=8, max_num_batched_tokens=4096,
            max_model_len=4096, kvcache_block_size=256)
        import nanovllm
        report["loaded_nanovllm_path"] = str(Path(nanovllm.__file__).resolve())
        require(Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()),
                "qualification imported production outside the selected checkout")
        report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16"}
        engine.configure_jetspec(args.draft, max_admissions_per_step=8)
        runtime = engine._jetspec_scheduler.runtime
        runtime.configure_optimizations(lightweight=True, batched_draft=True, feature_storage=True)
        specs = [replace(spec, arrival_step=0) for spec in shared.workload(prompts, 4, 8, 20)]
        actual_head_batches = []
        hook = runtime.head.register_forward_pre_hook(lambda module, args, kwargs:
            actual_head_batches.append(int(kwargs["noise_embedding"].shape[0])), with_kwargs=True)
        try:
            first = shared.serving_run(engine, specs, mode="jetspec", clock="step", label="early-c8")
        finally:
            hook.remove()
        report["early_c8"] = first
        report["early_c8_actual_head_batch_sizes"] = actual_head_batches
        require(all(request["status"] == "finished" for request in first["requests"]),
                "early optimized c8 did not finish")
        require(any(batch > 1 for batch in actual_head_batches), "early c8 had no genuine batched Draft forward")
        require(any(step["verification"] and step["verification"]["total_query_tokens"] == 408
                    for step in first["steps"]), "early c8 was not genuine Q408")
        save_json(args.output, report)
        print("EARLY optimized public c8 Q408 PASS", flush=True)
        original_assert = state_module.PagedTargetState.assert_round_invariant

        def strict(state, *a, **kwargs):
            kwargs["validate_device"] = True
            return original_assert(state, *a, **kwargs)

        original_verify = runtime._verify_batch
        checked_metadata = False

        def verify(requests, trees, transaction, metadata):
            nonlocal checked_metadata
            base = original_verify(requests, trees, transaction, metadata)
            if not checked_metadata and len(requests) == 8:
                masks = []
                for tree in trees:
                    parents = tree.host_parents
                    mask = torch.zeros((tree.num_nodes, tree.num_nodes), dtype=torch.bool, device="cpu")
                    for node in range(tree.num_nodes):
                        path = parent_chain(parents, node)
                        require(len(path) - 1 == tree.host_depths[node], "independent parent depth mismatch")
                        mask[node, path] = True
                    masks.append(mask.to(runtime.kv_pool.device))
                independent = PackedTreeMetadata.build(metadata.prefix_lengths, metadata.block_tables_host,
                    transaction.node_slots, masks, runtime.block_size, request_ids=metadata.request_ids)
                names = ("cu_seqlens_q", "query_to_request", "query_local_row", "prefix_lens", "block_tables",
                         "tree_slots", "qq_bias", "qq_bias_offsets", "node_counts")
                field_exact = {name: torch.equal(getattr(metadata, name), getattr(independent, name)) for name in names}
                comparison = original_verify(requests, trees, transaction, independent)
                check = {"Q": metadata.total_queries, "node_counts": list(metadata.node_counts_host),
                    "fields_bitwise_exact": field_exact, "target_logits_bitwise_exact": torch.equal(base[0], comparison[0]),
                    "target_taps_bitwise_exact": torch.equal(base[1], comparison[1])}
                require(all(field_exact.values()) and check["target_logits_bitwise_exact"] and
                        check["target_taps_bitwise_exact"], "fast metadata changed actual sameQ Target execution")
                report["metadata_checks"].append(check)
                checked_metadata = True
            return base

        with patch(state_module.PagedTargetState, "assert_round_invariant", strict), patch(runtime, "_verify_batch", verify):
            isolation = shared.eight_request_isolation(engine, prompts, args.draft)
            report["optimized_c8_target_isolation"] = isolation
            require(isolation["real_eight_request_packed_shape"] and isolation["finite_same_shape_controls_passed"] and
                    checked_metadata, "optimized c8 Target isolation/metadata failed")
            require(all(isolation["checks"]["all_layer_raw_copy_checks"]) and
                    all(isolation["checks"]["history_and_rejected_checks"]), "optimized all-layer commit/history failed")
            save_json(args.output, report)
            print("optimized c8 same-layout Target/metadata/raw commit/forced device checks PASS", flush=True)
            prompt = prompts["natural_language"]["prompt_token_ids"]
            for lengths, label in (((27, 43, 35), "cold-small-ragged"), ((27, 1024, 43), "cold-long-ragged"),
                                    ((1020, 1024, 1028), "near-1k-ragged")):
                requests = new_requests(runtime, prompt, lengths, label)
                try:
                    report["draft"].append(compare_draft(runtime, requests, label))
                    for request, delta in zip(requests, (1, 2, 3)):
                        request.drafter._fwd.cache.crop(request.state.cache_len - delta)
                    report["draft"].append(compare_draft(runtime, requests, label + "-warm"))
                    if label == "near-1k-ragged":
                        for request, delta in zip(requests, (1, 2, 3)):
                            request.drafter._fwd.cache.crop(request.state.cache_len - delta)
                        report["draft_isolation_and_padding"] = draft_isolation(runtime, requests)
                        newcomer = new_requests(runtime, prompt, (1022,), "newcomer")[0]
                        try:
                            report["draft"].append(compare_draft(runtime, [requests[2], newcomer, requests[0]],
                                                                "reorder-survivor-and-new-arrival"))
                        finally:
                            release_requests(runtime, [newcomer]) if len(runtime.requests) == 1 else runtime.cancel(newcomer)
                finally:
                    release_requests(runtime, requests)
                save_json(args.output, report)
                print(f"trained Draft {label} cold/warm envelope/history/compact caches PASS", flush=True)
            shared.set_mode(engine, "jetspec", args.draft, 2)
            with shared.ServingProbe(engine, diagnostic=True) as probe:
                first = shared.serving_run(engine, shared.qualification_workload(prompts), mode="jetspec",
                                           clock="step", label="lifecycle-a")
                checks = probe.metrics()
            second = shared.serving_run(engine, shared.qualification_workload(prompts), mode="jetspec",
                                        clock="step", label="lifecycle-b")
            exact = shared.replay_signature(first) == shared.replay_signature(second)
            report["lifecycle"] = {"first": first, "second": second, "checks": checks, "same_schedule_replay_exact": exact}
            require(exact and all(checks["all_layer_raw_copy_checks"]) and all(checks["history_and_rejected_checks"]),
                    "optimized lifecycle replay/copy/history failed")
            report["held_page_pressure"] = shared.allocator_pressure(engine, prompts)
            require(report["held_page_pressure"]["deferred_seen"] and report["held_page_pressure"]["recovered"],
                    "optimized held-page pressure failed")
            report["preemption"] = shared.preemption_qualification(engine, prompts)
            require(all(report["preemption"][key] for key in ("preemption_seen", "resume_seen",
                    "output_exactly_once", "fixed_schedule_replay_exact", "scratch_one_page_or_less")),
                    "optimized recompute preemption failed")
        require(engine.is_finished() and not runtime.requests, "qualification left requests live")
        engine.disable_jetspec()
        report["allocator_clean"] = not engine.scheduler.block_manager.used_block_ids
        require(report["allocator_clean"], "qualification leaked KV pages")
        after = qualification_identity(args)
        report["source_unchanged"] = after["production_source_sha256"] == source["production_source_sha256"]
        report["qualification_harness_unchanged"] = after["qualification_harness_source_sha256"] == source["qualification_harness_source_sha256"]
        require(report["source_unchanged"], "production changed during qualification")
        require(report["qualification_harness_unchanged"], "diagnostic code changed during qualification")
        report["all_gates_passed"] = True
        save_json(args.output, report)
        print("ALL trained-model Phase4 qualification gates PASS", flush=True)
    except BaseException as exception:
        report["error"] = f"{type(exception).__name__}: {exception}"
        save_json(args.output, report)
        raise
    finally:
        if engine is not None:
            serving = getattr(engine, "_jetspec_scheduler", None)
            if serving is not None:
                for request_id in list(serving.requests):
                    engine.cancel_request(request_id)
                serving.drain_events()
            runtime = getattr(engine, "_jetspec_batch_runtime", (None, None))[1]
            if runtime is not None:
                for request in list(runtime.requests.values()):
                    runtime.cancel(request)
            engine.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--target", default=shared.TARGET)
    parser.add_argument("--draft", default=shared.DRAFT)
    parser.add_argument("--oracle", default=shared.ORACLE)
    parser.add_argument("--output", default="/root/autodl-tmp/benchmarks/jetspec-phase4/qualification.json")
    run(parser.parse_args())
