#!/usr/bin/env python3
"""Diagnostic controls for trained-Qwen chunked prefill; no production changes.

The independent reference owns chronological dense K/V and constructs its own
offset mask. It invokes actual decoder.forward_dense, not a replica decoder.
FP32 controls cast only one decoder's parameters at a time and restore them.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

from jetspec_chunked_prefill import identity, checkpoint, prompt, raw_kv, require, save, tensor_metrics, bitwise_equal, argmax_witness


def independent_mask(prefix, count, device):
    import torch
    if prefix == 0:
        return None
    return torch.tensor([[key <= prefix + query for key in range(prefix + count)]
                         for query in range(count)], dtype=torch.bool, device=device)


@contextmanager
def fp32_parameters(module):
    import torch
    originals = [(parameter, parameter.data) for parameter in module.parameters()]
    try:
        for parameter, original in originals:
            parameter.data = original.to(torch.float32)
        yield
    finally:
        for parameter, original in originals:
            parameter.data = original


def chronological_forward(target, ids, plan, taps):
    import torch
    previous, outputs, features = None, [], []
    all_keys = [[] for _ in target.model.layers]
    all_values = [[] for _ in target.model.layers]
    cursor = 0
    for count in plan:
        positions = torch.arange(cursor, cursor + count, device=ids.device)
        hidden, new_kv, tapped = target.model.forward_dense(ids[cursor:cursor + count], positions,
            previous, independent_mask(cursor, count, ids.device), taps)
        outputs.append(hidden)
        features.append(tapped)
        for i, (key, value) in enumerate(new_kv):
            all_keys[i].append(key)
            all_values[i].append(value)
        previous = [(torch.cat(keys), torch.cat(values)) for keys, values in zip(all_keys, all_values)]
        cursor += count
    return torch.cat(outputs), torch.stack([torch.stack([torch.cat(keys), torch.cat(values)])
        for keys, values in zip(all_keys, all_values)], dim=1), torch.cat(features)


def compare_stage_records(full, chunks):
    from jetspec_layer_trace import tensor_metrics as stage_metrics
    require(list(full) == list(chunks), "diagnostic stage schema/order mismatch")
    metrics = OrderedDict((stage, stage_metrics(value, chunks[stage])) for stage, value in full.items())
    return {"earliest_nonbitwise_stage": next((stage for stage, metric in metrics.items()
             if not metric["bitwise_equal"]), None), "stages": metrics}


def layer_major(target, ids, plan, tapped_ids, *, fp32=False):
    import torch
    from nanovllm.models.qwen3 import _reference_rms_norm
    from jetspec_layer_trace import LayerTrace
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from contextlib import nullcontext
    full_hidden = target.model.embed_tokens(ids)
    chunk_hidden = full_hidden.clone()
    if fp32:
        full_hidden, chunk_hidden = full_hidden.float(), chunk_hidden.float()
    per_layer, full_keys, chunk_keys, full_taps, chunk_taps = [], [], [], [], []
    full_positions = torch.arange(ids.numel(), device=ids.device)
    for layer_id, layer in enumerate(target.model.layers):
        cast = fp32_parameters(layer) if fp32 else nullcontext()
        attention_backend = sdpa_kernel(SDPBackend.MATH) if fp32 else nullcontext()
        with cast, attention_backend:
            if fp32:
                full_output, _, full_kv, _ = layer.forward_dense(full_positions, full_hidden, None, None, None)
                full_records = None
            else:
                with LayerTrace(target, range(ids.numel())) as trace:
                    full_output, _, full_kv, _ = layer.forward_dense(full_positions, full_hidden, None, None, None)
                full_records = trace.records
            chunks, keys, values, chunk_records = [], [], [], OrderedDict()
            cursor = 0
            for count in plan:
                past = (torch.cat(keys), torch.cat(values)) if cursor else None
                args = (full_positions[cursor:cursor + count], chunk_hidden[cursor:cursor + count],
                        None, past, independent_mask(cursor, count, ids.device))
                if fp32:
                    output, _, kv, _ = layer.forward_dense(*args)
                else:
                    with LayerTrace(target, range(count)) as trace:
                        output, _, kv, _ = layer.forward_dense(*args)
                    for name, value in trace.records.items():
                        chunk_records.setdefault(name, []).append(value)
                chunks.append(output)
                keys.append(kv[0]); values.append(kv[1])
                cursor += count
            concatenated = torch.cat(chunks)
            chunk_kv = (torch.cat(keys), torch.cat(values))
            metrics = {"layer": layer_id,
                "input_hidden": tensor_metrics(chunk_hidden, full_hidden, 2e-4 if fp32 else 2**-6),
                "key": tensor_metrics(chunk_kv[0], full_kv[0], 2e-4 if fp32 else 2**-6),
                "value": tensor_metrics(chunk_kv[1], full_kv[1], 2e-4 if fp32 else 2**-6),
                "output_hidden": tensor_metrics(concatenated, full_output, 2e-4 if fp32 else 2**-6)}
            if not fp32:
                metrics["trace"] = compare_stage_records(full_records,
                    OrderedDict((name, torch.cat(values)) for name, values in chunk_records.items()))
            per_layer.append(metrics)
            full_keys.append(torch.stack(full_kv).cpu()); chunk_keys.append(torch.stack(chunk_kv).cpu())
            if layer_id in tapped_ids:
                full_taps.append(full_output.cpu()); chunk_taps.append(concatenated.cpu())
            full_hidden, chunk_hidden = full_output, concatenated
        if fp32:
            torch.cuda.empty_cache()
    with fp32_parameters(target.model.norm) if fp32 else nullcontext():
        full_hidden = _reference_rms_norm(full_hidden, target.model.norm)
        chunk_hidden = _reference_rms_norm(chunk_hidden, target.model.norm)
    with fp32_parameters(target.lm_head) if fp32 else nullcontext():
        full_logits = target.lm_head(full_hidden[-1:]).cpu()
        chunk_logits = target.lm_head(chunk_hidden[-1:]).cpu()
    return {"layers": per_layer, "final_hidden": tensor_metrics(chunk_hidden, full_hidden, 2e-4 if fp32 else 2**-6),
        "target_logits": tensor_metrics(chunk_logits, full_logits, 2e-4 if fp32 else 2**-6),
        "target_argmax": argmax_witness(chunk_logits, full_logits),
        "taps": tensor_metrics(torch.cat(chunk_taps, dim=-1), torch.cat(full_taps, dim=-1), 2e-4 if fp32 else 2**-6)}, {
        "full_kv": torch.stack(full_keys, dim=1), "chunk_kv": torch.stack(chunk_keys, dim=1),
        "full_taps": torch.cat(full_taps, dim=-1), "chunk_taps": torch.cat(chunk_taps, dim=-1)}


def run(args):
    if args.repo:
        sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch, nanovllm, jetspec
    import jetspec_layer_trace as trace_helper
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {"source": identity(nanovllm, jetspec), "arguments": vars(args),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trace_helper": {"path": str(Path(trace_helper.__file__).resolve()),
            "sha256": hashlib.sha256(Path(trace_helper.__file__).read_bytes()).hexdigest()},
        "models": {"target": checkpoint(args.target), "draft": checkpoint(args.draft)},
        "config": {"tensor_parallel_size": 1, "enforce_eager": True, "gpu_memory_utilization": .7,
            "max_num_batched_tokens": 4096, "max_model_len": 4096, "max_num_seqs": 2, "kvcache_block_size": 256},
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()},
        "contract": "same-shape paged versus independent chronological dense must be bitwise; FP32 controls diagnose, not relax failed BF16 bound",
        "fp32_control": {"matmul_allow_tf32": False, "sdpa_backend": "MATH", "roundoff_bound": 2e-4,
            "weights": "original BF16 weights promoted one decoder at a time; no whole-model FP32 allocation"}}
    engine, context, ready = None, None, None
    try:
        engine = nanovllm.LLM(args.target, tensor_parallel_size=1, enforce_eager=True,
            gpu_memory_utilization=.7, max_num_batched_tokens=4096, max_model_len=4096,
            max_num_seqs=2, kvcache_block_size=256)
        engine.configure_jetspec(args.draft, optimization="serving", enable_chunked_prefill=True,
            prefill_chunk_size=256, max_prefill_tokens=256)
        runtime = engine._jetspec_scheduler.runtime
        report["pool"] = {"shape": list(runtime.kv_pool.shape), "dtype": str(runtime.kv_pool.dtype)}
        tokens = prompt(runtime.tokenizer, args.length)
        plan = [args.chunk] * (args.length // args.chunk)
        if args.length % args.chunk:
            plan.append(args.length % args.chunk)
        ids = torch.tensor(tokens, dtype=torch.long, device=runtime.kv_pool.device)
        taps = runtime.target_layer_ids
        with torch.inference_mode():
            full_hidden, full_kv, full_taps = chronological_forward(runtime.target, ids, [args.length], taps)
            dense_hidden, dense_kv, dense_taps = chronological_forward(runtime.target, ids, plan, taps)
            context = runtime.begin_prefill(tokens, max_new_tokens=8, tree_budget=31, ignore_eos=True)
            for count in plan:
                ready = runtime.prefill_step(context, count)
            require(ready is not None, "paged chunk plan failed to promote")
            paged_kv = raw_kv(runtime, ready.state.logical_slots)
            report["same_shape_independent_reference"] = {
                "kv": tensor_metrics(paged_kv, dense_kv, 0),
                "taps": tensor_metrics(ready.state.target_hidden.squeeze(0), dense_taps, 0),
                "logical_prompt_equal": ready.state.committed[0, :-1].tolist() == tokens}
            require(all(report["same_shape_independent_reference"][k]["bitwise_equal"] for k in ("kv", "taps")),
                "paged differs from independent same-shape dense reference")
            report["bf16_chronological_vs_full"] = {
                "kv": tensor_metrics(dense_kv, full_kv, 2**-6),
                "taps": tensor_metrics(dense_taps, full_taps, 2**-6),
                "final_hidden": tensor_metrics(dense_hidden, full_hidden, 2**-6)}
            save(args.output, report)
            runtime.cancel(ready); ready = None
            report["bf16_layer_major"], payload = layer_major(runtime.target, ids, plan, taps)
            report["bf16_layer_major_independent_alignment"] = {
                "full_kv": bitwise_equal(payload["full_kv"], full_kv),
                "chunk_kv": bitwise_equal(payload["chunk_kv"], dense_kv),
                "full_taps": bitwise_equal(payload["full_taps"], full_taps),
                "chunk_taps": bitwise_equal(payload["chunk_taps"], dense_taps)}
            require(all(report["bf16_layer_major_independent_alignment"].values()), "layer-major diagnostic changed BF16 operands")
            save(args.output, report)
            del payload, full_hidden, full_kv, full_taps, dense_hidden, dense_kv, dense_taps, paged_kv
            torch.cuda.empty_cache()
            report["fp32_layer_major"], payload = layer_major(runtime.target, ids, plan, taps, fp32=True)
            control = report["fp32_layer_major"]
            require(all(metric["passed"] for layer in control["layers"] for metric in
                (layer["key"], layer["value"], layer["output_hidden"])) and
                all(control[k]["passed"] for k in ("final_hidden", "taps", "target_logits")), "FP32 control exceeded roundoff envelope")
            del payload
        engine.disable_jetspec()
        report["allocator_clean"] = not engine.scheduler.block_manager.used_block_ids
        report["source_unchanged"] = report["source"] == identity(nanovllm, jetspec)
        require(report["allocator_clean"] and report["source_unchanged"], "diagnostic cleanup/source stability failed")
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if engine is not None:
            serving = getattr(engine, "_jetspec_scheduler", None)
            if serving is not None:
                runtime = serving.runtime
                if ready is not None and ready.request_id in runtime.requests:
                    runtime.cancel(ready)
                if context is not None and context.request_id in runtime.prefills:
                    runtime.cancel_prefill(context)
            engine.exit()
        save(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True); parser.add_argument("--draft", required=True)
    parser.add_argument("--repo"); parser.add_argument("--length", type=int, default=33)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.length < 1 or args.chunk < 1:
        parser.error("length and chunk must be positive")
    run(args)


if __name__ == "__main__":
    main()
