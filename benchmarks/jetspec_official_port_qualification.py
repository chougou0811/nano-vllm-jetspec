#!/usr/bin/env python3
"""Fixed-gate full-model qualification of official-inspired Target execution.

Each live transaction is replayed from the same canonical state with reference
eager, candidate and candidate-repeat. Acceptance/commit occurs only afterwards.
No performance numbers from this diagnostic execution may be published.
"""
import argparse
import atexit
from pathlib import Path
import sys

import jetspec_tree_kernel_qualification as tree

frozen, chunks, serving = tree.frozen, tree.chunks, tree.serving


def run(args):
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch, nanovllm, jetspec
    from nanovllm import LLM
    from jetspec_final_upstream_flashattn import source_identity
    report = {"status": "in_progress", "passed": False, "invocation": vars(args),
              "source": source_identity(nanovllm, jetspec), "same_state": [],
              "contract": {"whole_network_bf16_bound": tree.BF16_BOUND,
                "attention_pre_store_fp32_bound": 1e-4,
                "cross_shape_token_bitwise_required": False,
                "request_isolation_byte_exact_required": True}}
    tree.save_json(args.output, report)
    engine = runtime = None
    try:
        torch.manual_seed(0)
        engine = LLM(args.target, enforce_eager=True, tensor_parallel_size=1,
            gpu_memory_utilization=.8, max_num_seqs=8, max_model_len=4096,
            max_num_batched_tokens=4096, kvcache_block_size=256)
        engine.configure_jetspec(args.draft, optimization="serving", attention_backend="sdpa",
            target_execution=args.target_execution, target_kernels=args.target_kernels, enable_chunked_prefill=False,
            max_prefill_tokens=4096, max_admissions_per_step=8)
        runtime = engine._jetspec_scheduler.runtime
        original_verify = runtime._verify_batch
        def kernel_policy(policy):
            model = runtime.target.model
            for layer in model.layers:
                layer.self_attn._jetspec_tree_fusion = policy != "reference"
        def replay(requests, trees, transaction, metadata):
            slots = torch.cat(transaction.node_slots)
            prefix = [chunks.raw_kv(runtime, r.state.logical_slots).clone() for r in requests]
            hidden_ref = []
            runtime._target_execution = "eager"
            kernel_policy("reference")
            hook = runtime.target.lm_head.register_forward_pre_hook(
                lambda module, inputs: hidden_ref.append(inputs[0].detach().clone()))
            try:
                ref_logits, ref_taps = original_verify(requests, trees, transaction, metadata)
            finally:
                hook.remove()
            ref_kv = chunks.raw_kv(runtime, slots).clone()
            runtime._target_execution = args.target_execution
            kernel_policy(args.target_kernels)
            hidden_candidate = []
            hook = None if args.target_execution == "cuda_graph" else runtime.target.lm_head.register_forward_pre_hook(
                lambda module, inputs: hidden_candidate.append(inputs[0].detach().clone()))
            try:
                logits, taps = original_verify(requests, trees, transaction, metadata)
            finally:
                if hook is not None:
                    hook.remove()
            hidden = runtime._target_graph.last_hidden if args.target_execution == "cuda_graph" else hidden_candidate[-1]
            hidden = hidden.clone()
            kv = chunks.raw_kv(runtime, slots).clone()
            logits, taps = logits.clone(), taps.clone()
            repeated_logits, repeated_taps = original_verify(requests, trees, transaction, metadata)
            repeat_kv = chunks.raw_kv(runtime, slots)
            check = {"prefix_lengths": list(metadata.prefix_lengths), "node_counts": list(metadata.node_counts_host),
                "all_layer_tree_kv": frozen.tensor_metrics(kv, ref_kv, bound=tree.BF16_BOUND),
                "final_hidden": frozen.tensor_metrics(hidden, hidden_ref[-1], bound=tree.BF16_BOUND),
                "target_taps": frozen.tensor_metrics(taps, ref_taps, bound=tree.BF16_BOUND),
                "lm_head": frozen.tensor_metrics(logits, ref_logits, bound=tree.BF16_BOUND),
                "argmax_flip_witness": tree.argmax_flip_witness(logits.cpu(), ref_logits.cpu()),
                "repeat_exact": all(tree.bits_equal(a, b) for a, b in
                    ((kv, repeat_kv), (logits, repeated_logits), (taps, repeated_taps))),
                "reference_candidate_exact": all(tree.bits_equal(a, b) for a, b in
                    ((kv, ref_kv), (logits, ref_logits), (taps, ref_taps), (hidden, hidden_ref[-1]))),
                "canonical_history_exact": all(tree.bits_equal(previous, chunks.raw_kv(runtime, request.state.logical_slots))
                    for previous, request in zip(prefix, requests))}
            if runtime._target_graph is not None:
                check["graph"] = runtime._target_graph.snapshot()
            check["passed"] = all(check[name]["passed"] for name in
                ("all_layer_tree_kv", "final_hidden", "target_taps", "lm_head")) and \
                check["repeat_exact"] and check["canonical_history_exact"] and \
                check["argmax_flip_witness"]["all_flip_witnesses_consistent"]
            report["same_state"].append(check)
            tree.save_json(args.output, report)
            frozen.require(check["passed"], "unchanged full-network/same-state numerical gate failed")
            return logits, taps
        prompt = runtime.tokenizer.encode("Explain why independent requests must not share tree ancestors.")
        requests = [runtime.create_request((prompt * ((n + len(prompt) - 1) // len(prompt)))[:n],
            max_new_tokens=32, tree_budget=(63, 31, 47)[i % 3], ignore_eos=True,
            request_id=f"port-{i}") for i, n in enumerate((33, 257, 1024, 2048, 65, 128, 255, 512))]
        with frozen.patch(runtime, "_verify_batch", replay), serving.ServingProbe(engine, diagnostic=True) as probe:
            for _ in range(2):
                runtime.step(requests)
            report["accepted_only_commit"] = probe.metrics()
        frozen.require(all(report["accepted_only_commit"]["all_layer_raw_copy_checks"]) and
            all(report["accepted_only_commit"]["history_and_rejected_checks"]), "raw accepted commit failed")
        for request in requests:
            runtime.cancel(request)
        runtime.release_idle_scratch()
        # Reuse the existing independent finite-poison isolation and complete
        # serving lifecycle gates; explicitly configure graph execution each time.
        def set_mode(selected_engine, mode, draft, concurrency):
            frozen.require(mode == "jetspec" and selected_engine is engine, "invalid qualification mode")
            engine.scheduler.max_num_seqs = concurrency
            engine.configure_jetspec(draft, optimization="serving", attention_backend="sdpa",
                target_execution=args.target_execution, target_kernels=args.target_kernels, enable_chunked_prefill=False,
                max_prefill_tokens=4096, max_admissions_per_step=2)
            engine._jetspec_scheduler.max_num_seqs = concurrency
        prompts = {row["id"]: row for row in __import__("json").loads(Path(args.oracle).read_text())["prompts"]}
        with frozen.patch(serving, "set_mode", set_mode):
            # The historical isolation probe assumes owned output tensors. Graph
            # outputs are borrowed and would be overwritten by its controls,
            # giving a vacuous comparison. Clone only in this diagnostic adapter.
            def owned_verify(*arguments):
                return tuple(value.clone() for value in original_verify(*arguments))
            with frozen.patch(runtime, "_verify_batch", owned_verify):
                report["isolation"] = serving.eight_request_isolation(engine, prompts, args.draft)
            frozen.require(report["isolation"]["finite_same_shape_controls_passed"], "request/ancestor isolation failed")
            set_mode(engine, "jetspec", args.draft, 2)
            with serving.ServingProbe(engine, diagnostic=True) as probe:
                first = serving.serving_run(engine, serving.qualification_workload(prompts),
                    mode="jetspec", clock="step", label="official-port-lifecycle", max_wall_s=args.deadline)
                report["lifecycle_raw_commit"] = probe.metrics()
            second = serving.serving_run(engine, serving.qualification_workload(prompts),
                mode="jetspec", clock="step", label="official-port-replay", max_wall_s=args.deadline)
            report["lifecycle"] = {"semantic_gates": tree.lifecycle_semantics(first, runtime.eos_token_ids),
                "fixed_schedule_replay_exact": serving.replay_signature(first) == serving.replay_signature(second)}
            frozen.require(report["lifecycle"]["fixed_schedule_replay_exact"], "delivery replay failed")
            report["pressure"] = serving.allocator_pressure(engine, prompts)
            report["preemption"] = serving.preemption_qualification(engine, prompts)
            frozen.require(all(report["pressure"][name] for name in ("deferred_seen", "recovered")), "pressure gate failed")
            frozen.require(all(report["preemption"][name] for name in
                ("preemption_seen", "resume_seen", "output_exactly_once", "fixed_schedule_replay_exact")), "preemption gate failed")
            def chunk_mode(selected_engine, selected_args, concurrency, chunk):
                engine.configure_jetspec(args.draft, optimization="serving", attention_backend="sdpa",
                    target_execution=args.target_execution, target_kernels=args.target_kernels,
                    enable_chunked_prefill=bool(chunk), prefill_chunk_size=chunk or 256,
                    max_prefill_tokens=chunk or 4096)
                engine._jetspec_scheduler.max_num_seqs = concurrency
                return runtime
            with frozen.patch(chunks, "configure", chunk_mode):
                report["chunked_lifecycle_recompute"] = chunks.lifecycle_qualification(engine, args)
        report["graph"] = runtime._target_graph.snapshot() if runtime._target_graph else None
        engine.disable_jetspec()
        report["allocator_cleanup"] = not engine.scheduler.block_manager.used_block_ids
        frozen.require(report["allocator_cleanup"], "allocator cleanup failed")
        report.update(status="complete", passed=True, source_end=source_identity(nanovllm, jetspec))
        tree.save_json(args.output, report)
    except BaseException as error:
        report.update(status="failed", passed=False, failure={"type": type(error).__name__, "message": str(error)})
        tree.save_json(args.output, report)
        raise
    finally:
        if engine is not None:
            atexit.unregister(engine.exit)
            engine.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "target", "draft", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--target-execution", choices=("eager", "cuda_graph"), default="cuda_graph")
    parser.add_argument("--target-kernels", choices=("reference", "fused_rope"), default="reference")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--oracle", default=serving.ORACLE)
    parser.add_argument("--deadline", type=float, default=600)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
