"""JetSpec-style paged-tree attention with a single-request FP32 accumulator.

The fallback launch contract follows pinned JetSpec commit 2c7b3fa. The
single-request specialization preserves its logical-slot/ancestor semantics.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _single_request_paged_tree_fp32(
    out_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    slots_ptr,
    qq_bias_ptr,
    seq_len,
    query_len: tl.constexpr,
    scale,
    q_stride_0,
    q_stride_1,
    out_stride_0,
    out_stride_1,
    bias_stride_0,
    k_stride_0,
    k_stride_1,
    k_stride_2,
    k_stride_3: tl.constexpr,
    v_stride_0,
    v_stride_1,
    v_stride_2,
    v_stride_3: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILE: tl.constexpr,
):
    """Single-request specialization of JetSpec's paged online-softmax contract."""
    query_idx = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_head = query_head // num_queries_per_kv
    d = tl.arange(0, BLOCK_D)
    dmask = d < head_size
    q = tl.load(
        q_ptr + query_idx * q_stride_0 + query_head * q_stride_1 + d,
        mask=dmask,
        other=0.0,
    ).to(tl.float32)
    context_len = seq_len - query_len
    running_max = tl.full((), float("-inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    num_tiles = tl.cdiv(seq_len, TILE)
    for tile_idx in range(0, num_tiles):
        key_pos = tile_idx * TILE + tl.arange(0, TILE)
        valid = key_pos < seq_len
        slot = tl.load(slots_ptr + key_pos, mask=valid, other=0).to(tl.int64)
        physical_block = slot // block_size
        physical_offset = slot % block_size
        k = tl.load(
            k_ptr
            + physical_block[:, None] * k_stride_0
            + physical_offset[:, None] * k_stride_1
            + kv_head * k_stride_2
            + d[None, :] * k_stride_3,
            mask=valid[:, None] & dmask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        key_rel = key_pos - context_len
        is_node = (key_rel >= 0) & (key_rel < query_len)
        bias = tl.load(
            qq_bias_ptr + query_idx * bias_stride_0 + key_rel,
            mask=valid & is_node,
            other=0.0,
        ).to(tl.float32)
        visible = valid & (key_pos <= context_len + query_idx)
        scores = tl.where(visible, scores + bias, float("-inf"))
        tile_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, tile_max)
        next_max = tl.where(next_max > float("-inf"), next_max, 0.0)
        alpha = tl.exp(running_max - next_max)
        probability = tl.exp(scores - next_max)
        v = tl.load(
            v_ptr
            + physical_block[:, None] * v_stride_0
            + physical_offset[:, None] * v_stride_1
            + kv_head * v_stride_2
            + d[None, :] * v_stride_3,
            mask=valid[:, None] & dmask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * alpha + tl.sum(probability[:, None] * v, axis=0)
        running_sum = running_sum * alpha + tl.sum(probability, axis=0)
        running_max = next_max
    output = accumulator / running_sum
    tl.store(
        out_ptr + query_idx * out_stride_0 + query_head * out_stride_1 + d,
        output,
        mask=dmask,
    )


def paged_tree_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_k: torch.Tensor,
    qq_bias: torch.Tensor | None,
    scale: float,
    num_queries_per_kv: int,
    block_size: int,
    logical_kv_slots: torch.Tensor | None = None,
    logical_kv_starts: torch.Tensor | None = None,
    logical_kv_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    total_q, num_query_heads, head_size = q.shape
    num_seqs = int(seq_lens_k.shape[0])
    if num_seqs == 1 and logical_kv_slots is not None and qq_bias is not None:
        out = torch.empty_like(q)
        seq_len = int(seq_lens_k[0].item())
        tile = 64
        _single_request_paged_tree_fp32[(total_q, num_query_heads)](
            out,
            q,
            k_pool,
            v_pool,
            logical_kv_slots.view(-1),
            qq_bias,
            seq_len,
            query_len=total_q,
            scale=scale,
            q_stride_0=q.stride(0),
            q_stride_1=q.stride(1),
            out_stride_0=out.stride(0),
            out_stride_1=out.stride(1),
            bias_stride_0=qq_bias.stride(0),
            k_stride_0=k_pool.stride(0),
            k_stride_1=k_pool.stride(1),
            k_stride_2=k_pool.stride(2),
            k_stride_3=k_pool.stride(3),
            v_stride_0=v_pool.stride(0),
            v_stride_1=v_pool.stride(1),
            v_stride_2=v_pool.stride(2),
            v_stride_3=v_pool.stride(3),
            num_queries_per_kv=num_queries_per_kv,
            block_size=block_size,
            head_size=head_size,
            BLOCK_D=triton.next_power_of_2(head_size),
            TILE=tile,
        )
        return out

    from jetspec.inference_engine.paged_tree_attn import _kernel_paged_tree_attn
    out = torch.empty_like(q)
    use_qq_bias = qq_bias is not None
    use_logical_slots = logical_kv_slots is not None
    if use_logical_slots and (logical_kv_starts is None or logical_kv_lens is None):
        raise ValueError("logical KV slots require starts and lengths")

    block_m = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    block_q = block_m // num_queries_per_kv
    total_num_q_blocks = (total_q + block_q - 1) // block_q
    num_kv_heads = int(k_pool.shape[2])
    tile_size = 128
    _kernel_paged_tree_attn[(total_num_q_blocks, num_kv_heads)](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k_pool,
        value_cache_ptr=v_pool,
        block_tables_ptr=block_table,
        logical_kv_slots_ptr=logical_kv_slots,
        logical_kv_starts_ptr=logical_kv_starts,
        logical_kv_lens_ptr=logical_kv_lens,
        seq_lens_ptr=seq_lens_k,
        qq_bias_ptr=qq_bias,
        scale=scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        logical_kv_slots_stride=(logical_kv_slots.stride(0) if use_logical_slots else 0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_QQ_BIAS=use_qq_bias,
        USE_LOGICAL_KV_SLOTS=use_logical_slots,
        stride_k_cache_0=k_pool.stride(0),
        stride_k_cache_1=k_pool.stride(1),
        stride_k_cache_2=k_pool.stride(2),
        stride_k_cache_3=k_pool.stride(3),
        stride_v_cache_0=v_pool.stride(0),
        stride_v_cache_1=v_pool.stride(1),
        stride_v_cache_2=v_pool.stride(2),
        stride_v_cache_3=v_pool.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        num_seqs=num_seqs,
        BLOCK_M=block_m,
    )
    return out
