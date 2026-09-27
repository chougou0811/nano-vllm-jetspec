import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:  # correctness fallback for environments without a matching wheel
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    @staticmethod
    def _repeat_kv(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
        groups = num_q_heads // x.shape[1]
        return x.repeat_interleave(groups, dim=1) if groups > 1 else x

    def _sdpa_one(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  prefix_len: int) -> torch.Tensor:
        """Correctness fallback for one sequence; tensors are (T, heads, dim)."""
        q_len, k_len = q.shape[0], k.shape[0]
        k = self._repeat_kv(k, q.shape[1])
        v = self._repeat_kv(v, q.shape[1])
        qh = q.transpose(0, 1).unsqueeze(0)
        kh = k.transpose(0, 1).unsqueeze(0)
        vh = v.transpose(0, 1).unsqueeze(0)
        qi = torch.arange(q_len, device=q.device).unsqueeze(1)
        kj = torch.arange(k_len, device=q.device).unsqueeze(0)
        allowed = kj <= (prefix_len + qi)
        out = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=allowed.view(1, 1, q_len, k_len),
            dropout_p=0.0, scale=self.scale,
        )
        return out.squeeze(0).transpose(0, 1).contiguous()

    @staticmethod
    def _gather_paged(cache: torch.Tensor, block_table: torch.Tensor,
                      length: int) -> torch.Tensor:
        block_size = cache.shape[1]
        pos = torch.arange(length, device=cache.device)
        blocks = block_table[pos // block_size].long()
        return cache[blocks, pos % block_size]

    def _torch_sdpa_fallback(self, q: torch.Tensor, k: torch.Tensor,
                             v: torch.Tensor, context) -> torch.Tensor:
        """Slow eager fallback. FlashAttention remains the normal path when installed."""
        outputs = []
        if context.is_prefill:
            for i in range(context.cu_seqlens_q.numel() - 1):
                q0 = int(context.cu_seqlens_q[i].item())
                q1 = int(context.cu_seqlens_q[i + 1].item())
                k0 = int(context.cu_seqlens_k[i].item())
                k1 = int(context.cu_seqlens_k[i + 1].item())
                if context.block_tables is None:
                    ks, vs = k[k0:k1], v[k0:k1]
                else:
                    ks = self._gather_paged(self.k_cache, context.block_tables[i], k1 - k0)
                    vs = self._gather_paged(self.v_cache, context.block_tables[i], k1 - k0)
                outputs.append(self._sdpa_one(q[q0:q1], ks, vs, (k1 - k0) - (q1 - q0)))
        else:
            for i in range(q.shape[0]):
                length = int(context.context_lens[i].item())
                ks = self._gather_paged(self.k_cache, context.block_tables[i], length)
                vs = self._gather_paged(self.v_cache, context.block_tables[i], length)
                outputs.append(self._sdpa_one(q[i:i + 1], ks, vs, length - 1))
        return torch.cat(outputs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if flash_attn_varlen_func is None or flash_attn_with_kvcache is None:
            return self._torch_sdpa_fallback(q, k, v, context)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
