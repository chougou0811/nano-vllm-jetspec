#!/usr/bin/env python3
"""Opt-in FlashAttention GPU qualification; no model/production modifications."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time


def reference(q, k, v, causal):
    import torch
    q, k, v = q.float(), k.float(), v.float()
    k, v = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
    if causal:
        # Bottom-right causal alignment (including the decode query case).
        rows = torch.arange(q.shape[0], device=q.device)[:, None]
        cols = torch.arange(k.shape[0], device=k.device)[None, :]
        scores.masked_fill_(cols > rows + k.shape[0] - q.shape[0], float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def run(args):
    import torch
    import flash_attn
    import flash_attn_2_cuda
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    report = {"status": "running", "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
              "flash_attn": importlib.metadata.version("flash-attn"),
              "package_path": flash_attn.__file__, "extension_path": flash_attn_2_cuda.__file__,
              "extension_sha256": hashlib.sha256(Path(flash_attn_2_cuda.__file__).read_bytes()).hexdigest(),
              "varlen_callable": callable(flash_attn_varlen_func),
              "kvcache_callable": callable(flash_attn_with_kvcache),
              "no_sdpa_reference": True, "model_loaded": False, "benchmark_executed": False,
              "reference": "Independent FP32 matmul/softmax, BF16 input, max absolute error <=0.02",
              "cases": []}
    started = time.perf_counter()
    torch.manual_seed(2026)
    try:
        with torch.inference_mode():
            for lengths in ((17, 31), (128, 256), (1024, 2048)):
                total = sum(lengths)
                q = torch.randn(total, 32, 128, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(total, 8, 128, device="cuda", dtype=torch.bfloat16)
                v = torch.randn_like(k)
                cu = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
                out = flash_attn_varlen_func(q, k, v, cu, cu, max(lengths), max(lengths), causal=True)
                torch.cuda.synchronize()
                errors, offset = [], 0
                for n in lengths:
                    expected = reference(q[offset:offset+n], k[offset:offset+n], v[offset:offset+n], True)
                    errors.append((out[offset:offset+n].float() - expected).abs().max().item())
                    offset += n
                assert max(errors) <= .02, ("varlen", lengths, errors)
                report["cases"].append({"api": "flash_attn_varlen_func", "lengths": list(lengths),
                                        "gpu_call_passed": True, "max_abs_errors": errors})

                b = 256
                per_seq_pages = (max(lengths) + b - 1) // b
                blocks = 2 * per_seq_pages
                kc = torch.zeros(blocks, b, 8, 128, device="cuda", dtype=torch.bfloat16)
                vc = torch.zeros_like(kc)
                # Reverse each request's physical-page range to exercise a real page table.
                table = torch.tensor([list(range(per_seq_pages, blocks)), list(range(per_seq_pages))],
                                     device="cuda", dtype=torch.int32)
                offset = 0
                for i, n in enumerate(lengths):
                    physical = list(range(per_seq_pages, blocks)) if i == 0 else list(range(per_seq_pages))
                    for logical, page in enumerate(physical):
                        count = max(0, min(b, n - logical*b))
                        if count:
                            kc[page, :count] = k[offset+logical*b:offset+logical*b+count]
                            vc[page, :count] = v[offset+logical*b:offset+logical*b+count]
                    offset += n
                dq = torch.stack((q[lengths[0]-1], q[-1]))[:, None]
                lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
                decoded = flash_attn_with_kvcache(dq, kc, vc, cache_seqlens=lens, block_table=table, causal=True)
                torch.cuda.synchronize()
                errors, offset = [], 0
                for i, n in enumerate(lengths):
                    expected = reference(dq[i], k[offset:offset+n], v[offset:offset+n], True)
                    errors.append((decoded[i].float() - expected).abs().max().item())
                    offset += n
                assert max(errors) <= .02, ("kvcache", lengths, errors)
                report["cases"].append({"api": "flash_attn_with_kvcache", "lengths": list(lengths),
                                        "page_size": b, "nontrivial_page_table": True,
                                        "gpu_call_passed": True, "max_abs_errors": errors})
        report.update(status="passed", passed=True)
    except BaseException as error:
        report.update(status="failed", passed=False, error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["wall_s"] = time.perf_counter() - started
        report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
