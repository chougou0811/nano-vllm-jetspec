"""Explicit external FlashAttention for causal Target prefill/recompute.

The caller supplies causal *semantics*, not an arbitrary tree mask. Queries
are the suffix of the logical K/V sequence, so FlashAttention's bottom-right
causal alignment implements both full prefill and offset chunked prefill.
Packed tree verification deliberately does not use this BF16 backend.
"""
from dataclasses import dataclass
from functools import lru_cache

import torch
from packaging.version import Version


@lru_cache(maxsize=1)
def require_flash_attention():
    try:
        import flash_attn
        from flash_attn import flash_attn_varlen_func
    except ImportError as exc:
        raise RuntimeError(
            "attention_backend='flash_attn' requires a working external "
            "FlashAttention CUDA installation; no SDPA fallback is permitted"
        ) from exc
    if Version(flash_attn.__version__) < Version("2.1"):
        raise RuntimeError("Flash prefill requires flash-attn >= 2.1 for bottom-right causal alignment")
    return flash_attn_varlen_func


@dataclass(frozen=True)
class FlashPrefillMetadata:
    cu_query: torch.Tensor
    cu_key: torch.Tensor
    query_length: int
    key_length: int

    @classmethod
    def build(cls, query_length, key_length, device):
        if query_length < 1 or key_length < query_length:
            raise ValueError("causal prefill requires 0 < query length <= key length")
        cu_query = torch.tensor([0, query_length], dtype=torch.int32, device=device)
        cu_key = cu_query if key_length == query_length else torch.tensor(
            [0, key_length], dtype=torch.int32, device=device)
        return cls(cu_query, cu_key, int(query_length), int(key_length))


def flash_causal_prefill(q, k, v, metadata, scale):
    if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape:
        raise ValueError("Flash prefill requires flat (tokens, heads, dim) Q/K/V")
    if (q.shape[0] != metadata.query_length or k.shape[0] != metadata.key_length
            or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]):
        raise ValueError("Flash prefill metadata/GQA geometry mismatch")
    if (not q.is_cuda or q.device != k.device or q.device != v.device
            or q.device != metadata.cu_query.device or q.device != metadata.cu_key.device
            or q.dtype not in (torch.float16, torch.bfloat16)
            or q.dtype != k.dtype or q.dtype != v.dtype):
        raise ValueError("Flash prefill requires FP16/BF16 Q/K/V on one CUDA device")
    return require_flash_attention()(
        q, k, v, metadata.cu_query, metadata.cu_key,
        metadata.query_length, metadata.key_length,
        dropout_p=0.0, softmax_scale=scale, causal=True,
    )
