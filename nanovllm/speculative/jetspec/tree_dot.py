"""Experimental Tensor Core paged ragged Tree Attention.

Unlike the scalar reference, QK/PV reduction order changes in this experiment.
BF16/FP16 QK retains the original operands and uses FP32 dot accumulation; FP32
QK uses tf32x3. Softmax probabilities stay FP32, and PV uses FP32 P/V with
explicit tf32x3 input precision -- probabilities are NEVER rounded to BF16.
Request-local tiling and ancestor visibility are unchanged. This entry has no
production dispatch; it must pass the existing pre-store FP32 numerical gates.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from nanovllm.speculative.jetspec.tree_attention import (
    _validate,
    packed_tree_attention_reference,
)


@triton.jit
def _packed_tree_dot_fp32(
    out_ptr, q_ptr, k_ptr, v_ptr,
    prefix_lens_ptr, node_counts_ptr, cu_query_ptr,
    block_tables_ptr, tree_slots_ptr, bias_ptr, bias_offsets_ptr,
    scale,
    q_stride_0, q_stride_1, out_stride_0, out_stride_1, table_stride_0,
    k_stride_0, k_stride_1, k_stride_2, k_stride_3: tl.constexpr,
    v_stride_0, v_stride_1, v_stride_2, v_stride_3: tl.constexpr,
    GROUPS: tl.constexpr, QUERY_TILE: tl.constexpr, ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr, TILE: tl.constexpr, Q_FP32: tl.constexpr,
):
    query_block = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.program_id(2)
    context_len = tl.load(prefix_lens_ptr + request)
    node_count = tl.load(node_counts_ptr + request)
    tree_start = tl.load(cu_query_ptr + request)
    bias_start = tl.load(bias_offsets_ptr + request)
    rows = tl.arange(0, ROWS)
    local_q = query_block * QUERY_TILE + rows // GROUPS
    query_head = kv_head * GROUPS + rows % GROUPS
    row_valid = (rows < QUERY_TILE * GROUPS) & (local_q < node_count)
    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < HEAD_SIZE
    # Keep BF16/FP16 source operands: do not upcast/cast them through TF32.
    q = tl.load(q_ptr + (tree_start + local_q[:, None]) * q_stride_0
                + query_head[:, None] * q_stride_1 + dims[None, :],
                mask=row_valid[:, None] & dim_valid[None, :], other=0.0)
    running_max = tl.full((ROWS,), float("-inf"), tl.float32)
    running_sum = tl.zeros((ROWS,), tl.float32)
    accumulator = tl.zeros((ROWS, BLOCK_D), tl.float32)
    seq_len = context_len + node_count
    for tile in range(0, tl.cdiv(seq_len, TILE)):
        positions = tile * TILE + tl.arange(0, TILE)
        valid = positions < seq_len
        prefix = positions < context_len
        node = positions - context_len
        in_tree = (node >= 0) & (node < node_count)
        page = tl.load(block_tables_ptr + request * table_stride_0 + positions // BLOCK_SIZE,
                       mask=valid & prefix, other=0).to(tl.int64)
        scratch = tl.load(tree_slots_ptr + tree_start + node,
                          mask=valid & in_tree, other=0).to(tl.int64)
        slots = tl.where(prefix, page * BLOCK_SIZE + positions % BLOCK_SIZE, scratch)
        addresses, offsets = slots // BLOCK_SIZE, slots % BLOCK_SIZE
        k = tl.load(k_ptr + addresses[:, None] * k_stride_0
                    + offsets[:, None] * k_stride_1 + kv_head * k_stride_2
                    + dims[None, :] * k_stride_3,
                    mask=valid[:, None] & dim_valid[None, :], other=0.0)
        if Q_FP32:
            scores = tl.dot(q, tl.trans(k), input_precision="tf32x3",
                            out_dtype=tl.float32) * scale
        else:
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        bias = tl.load(bias_ptr + bias_start + local_q[:, None] * node_count + node[None, :],
                       mask=row_valid[:, None] & valid[None, :] & in_tree[None, :],
                       other=0.0).to(tl.float32)
        visible = (row_valid[:, None] & valid[None, :]
                   & (positions[None, :] <= context_len + local_q[:, None]))
        scores = tl.where(visible, scores + bias, float("-inf"))
        tile_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, tile_max)
        next_max = tl.where(next_max > float("-inf"), next_max, 0.0)
        alpha = tl.exp(running_max - next_max)
        probability = tl.exp(scores - next_max[:, None])
        v = tl.load(v_ptr + addresses[:, None] * v_stride_0
                    + offsets[:, None] * v_stride_1 + kv_head * v_stride_2
                    + dims[None, :] * v_stride_3,
                    mask=valid[:, None] & dim_valid[None, :], other=0.0).to(tl.float32)
        # Explicit FP32 probabilities + FP32 V, triple-TF32 multiplication.
        # No FP16/BF16 probability path is implemented or silently selected.
        pv = tl.dot(probability.to(tl.float32), v, input_precision="tf32x3",
                    out_dtype=tl.float32)
        accumulator = accumulator * alpha[:, None] + pv
        running_sum = running_sum * alpha + tl.sum(probability, axis=1)
        running_max = next_max
    output = accumulator / tl.where(row_valid, running_sum, 1.0)[:, None]
    tl.store(out_ptr + (tree_start + local_q[:, None]) * out_stride_0
             + query_head[:, None] * out_stride_1 + dims[None, :], output,
             mask=row_valid[:, None] & dim_valid[None, :])


def packed_tree_attention_dot(q, k_pool, v_pool, metadata, scale,
                              num_queries_per_kv, *, output_dtype=None,
                              query_tile=4, num_warps=4):
    """Explicit Tensor Core precision experiment; no default serving selection.

    Non-power-of-two GQA uses the qualified scalar reference. ROWS is padded to
    at least 16 for Tensor Core dot geometry; padding never crosses requests or
    produces extra valid query rows. The online softmax key tile remains 64.
    """
    if isinstance(query_tile, bool) or query_tile not in (1, 2, 4, 8):
        raise ValueError("query_tile must be 1, 2, 4 or 8")
    if isinstance(num_warps, bool) or num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8 or 16")
    if (isinstance(num_queries_per_kv, int) and not isinstance(num_queries_per_kv, bool)
            and num_queries_per_kv > 0
            and num_queries_per_kv & (num_queries_per_kv - 1)):
        return packed_tree_attention_reference(q, k_pool, v_pool, metadata, scale,
                                               num_queries_per_kv,
                                               output_dtype=output_dtype)
    out = _validate(q, k_pool, v_pool, metadata, num_queries_per_kv, output_dtype)
    grid = (triton.cdiv(max(metadata.node_counts_host), query_tile),
            k_pool.shape[2], metadata.num_requests)
    _packed_tree_dot_fp32[grid](
        out, q, k_pool, v_pool, metadata.prefix_lens, metadata.node_counts,
        metadata.cu_seqlens_q, metadata.block_tables, metadata.tree_slots,
        metadata.qq_bias, metadata.qq_bias_offsets, scale,
        q_stride_0=q.stride(0), q_stride_1=q.stride(1),
        out_stride_0=out.stride(0), out_stride_1=out.stride(1),
        table_stride_0=metadata.block_tables.stride(0),
        k_stride_0=k_pool.stride(0), k_stride_1=k_pool.stride(1),
        k_stride_2=k_pool.stride(2), k_stride_3=k_pool.stride(3),
        v_stride_0=v_pool.stride(0), v_stride_1=v_pool.stride(1),
        v_stride_2=v_pool.stride(2), v_stride_3=v_pool.stride(3),
        GROUPS=num_queries_per_kv, QUERY_TILE=query_tile,
        ROWS=max(16, triton.next_power_of_2(query_tile * num_queries_per_kv)),
        BLOCK_SIZE=metadata.block_size, HEAD_SIZE=q.shape[-1],
        BLOCK_D=max(32, triton.next_power_of_2(q.shape[-1])), TILE=64,
        Q_FP32=q.dtype == torch.float32, num_warps=num_warps,
    )
    return out
