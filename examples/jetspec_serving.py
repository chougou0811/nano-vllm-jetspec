#!/usr/bin/env python3
"""Portable, bounded serving smoke: no private paths, oracle, or remote code.

Install this checkout first; invoke --help without model weights or CUDA.
This is a smoke test, not a throughput benchmark or strict AR-token oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path,
                        help="local Qwen/Qwen3-8B checkpoint directory")
    parser.add_argument("--draft", required=True, type=Path,
                        help="local JetSpec/jetspec-qwen3-8b checkpoint directory")
    parser.add_argument("--prompt", action="append", help="repeat for multiple requests")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-prefill-tokens", type=int, default=None)
    parser.add_argument("--chunked-prefill", action=argparse.BooleanOptionalAction, default=True,
                        help="interleave prefill/recompute chunks with speculative rounds")
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="fail instead of looping forever on an invalid setup")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75,
                        help="Target KV pool fraction; leave space for the Draft and features")
    parser.add_argument("--output", type=Path, help="optional machine-readable smoke report")
    args = parser.parse_args(argv)
    for name in ("target", "draft"):
        path = getattr(args, name)
        if not path.is_dir() or not (path / "config.json").is_file():
            parser.error(f"--{name} must be a local checkpoint directory containing config.json")
    if args.max_tokens < 1 or args.max_model_len < 2 or args.max_steps < 1:
        parser.error("require max-tokens >= 1, max-model-len >= 2, max-steps >= 1")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must lie strictly between 0 and 1")
    if args.max_prefill_tokens is not None and args.max_prefill_tokens < 1:
        parser.error("max-prefill-tokens must be positive")
    if args.prefill_chunk_size < 1:
        parser.error("prefill-chunk-size must be positive")
    return args


def run(args):
    import torch
    from nanovllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("JetSpec serving requires an NVIDIA CUDA GPU; run tools/check_environment.py first")
    prompts = args.prompt or ["What is the capital of France? Answer briefly.",
                              "Explain why the sky appears blue in one sentence."]
    engine = LLM(str(args.target.resolve()), enforce_eager=True, tensor_parallel_size=1,
                 max_num_seqs=max(1, len(prompts)), max_model_len=args.max_model_len,
                 max_num_batched_tokens=args.max_model_len,
                 gpu_memory_utilization=args.gpu_memory_utilization)
    outputs = {f"request-{index}": [] for index in range(len(prompts))}
    terminal = {}
    try:
        engine.configure_jetspec(str(args.draft.resolve()),
                                 max_prefill_tokens=args.max_prefill_tokens,
                                 enable_chunked_prefill=args.chunked_prefill,
                                 prefill_chunk_size=args.prefill_chunk_size)
        for index, prompt in enumerate(prompts):
            ids = engine.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            # The supported trained head drafts 15 lookahead positions even
            # when the requested output cap is smaller than its whole tree.
            if len(ids) + args.max_tokens + 15 > args.max_model_len:
                raise ValueError("prompt + requested output + 15 tree lookahead exceeds --max-model-len")
            engine.add_request(ids, SamplingParams(temperature=0, max_tokens=args.max_tokens),
                               request_id=f"request-{index}", tree_budget=(63, 31, 47)[index % 3])
        steps = 0
        while not engine.is_finished():
            if steps >= args.max_steps:
                raise RuntimeError("smoke exceeded --max-steps")
            engine.step()
            steps += 1
            info = engine.last_step_info
            for event in info["events"]:
                key = event["request_id"]
                if event["kind"] == "tokens":
                    outputs[key].extend(event["token_ids"])
                else:
                    if key in terminal:
                        raise AssertionError("duplicate terminal event")
                    terminal[key] = event
                    if event["kind"] != "finished":
                        raise RuntimeError(f"request failed: {event}")
                    if outputs[key] != event["token_ids"]:
                        raise AssertionError("incremental delivery differs from terminal output")
            if info.get("blocked"):
                raise RuntimeError(f"smoke cannot progress: {info.get('blocked_reason')}")
        if set(terminal) != set(outputs):
            raise AssertionError("not every submitted request produced a terminal event")
        result = {
            "kind": "serving_smoke_not_benchmark", "steps": steps,
            "torch": torch.__version__, "target": str(args.target.resolve()),
            "draft": str(args.draft.resolve()), "exactly_once_delivery": True,
            "requests": [{"request_id": key, "token_ids": ids,
                          "text": engine.tokenizer.decode(ids, skip_special_tokens=True)}
                         for key, ids in outputs.items()],
        }
        engine.disable_jetspec()
        return result
    finally:
        engine.exit()


def main(argv=None):
    args = arguments(argv)
    result = run(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
