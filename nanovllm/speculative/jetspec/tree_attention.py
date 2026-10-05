"""Standalone FP32 paged ragged tree attention operators.

The reference preserves Phase 3.1's one-(query, head) kernel. The reuse
operator groups GQA heads and optionally adjacent query rows *within one
request*, loading their shared K/V tile once per program. Both use FP32
multiply/reduction and TILE=64 online softmax; neither rounds probabilities
to BF16 or relies on flat causal masking instead of ancestor visibility.

These explicit entries also support FP32 output with unchanged BF16 operands
for pre-store numerical qualification. Production dispatch is separate.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata


def _validate(q, k, v, metadata, groups, output_dtype):
    if q.ndim != 3 or q.shape[0] != metadata.total_queries:
        raise ValueError("packed queries do not match the metadata")
    if k.ndim != 4 or k.shape != v.shape or k.shape[1] != metadata.block_size:
        raise ValueError("packed K/V page geometry does not match metadata")
    if not q.is_cuda or q.device != k.device or q.device != v.device or q.device != metadata.tree_slots.device:
        raise ValueError("packed attention requires CUDA queries, KV and metadata on one device")
    if q.stride(-1) != 1 or k.shape[-1] != q.shape[-1]:
        raise ValueError("unsupported packed query/KV head geometry")
    if isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0 or q.shape[1] != k.shape[2] * groups:
        raise ValueError("invalid packed GQA head grouping")
    if q.dtype != k.dtype or k.dtype != v.dtype or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("packed Q/K/V must use one supported floating dtype")
    dtype = q.dtype if output_dtype is None else output_dtype
    if dtype not in (q.dtype, torch.float32):
        raise ValueError("output dtype must match input or be FP32 for qualification")
    return torch.empty(q.shape, dtype=dtype, device=q.device)


def packed_tree_attention_reference(q, k_pool, v_pool, metadata, scale,
                                    num_queries_per_kv, *, output_dtype=None):
    """Frozen scalar-query operator, optionally exposing FP32 pre-store output."""
    from nanovllm.speculative.jetspec.paged_backend import _packed_paged_tree_fp32

    out = _validate(q, k_pool, v_pool, metadata, num_queries_per_kv, output_dtype)
    _packed_paged_tree_fp32[(metadata.total_queries, q.shape[1])](
        out, q, k_pool, v_pool,
        metadata.query_to_request, metadata.query_local_row,
        metadata.prefix_lens, metadata.node_counts, metadata.cu_seqlens_q,
        metadata.block_tables, metadata.tree_slots, metadata.qq_bias, metadata.qq_bias_offsets,
        scale,
        q_stride_0=q.stride(0), q_stride_1=q.stride(1),
        out_stride_0=out.stride(0), out_stride_1=out.stride(1),
        table_stride_0=metadata.block_tables.stride(0),
        k_stride_0=k_pool.stride(0), k_stride_1=k_pool.stride(1),
        k_stride_2=k_pool.stride(2), k_stride_3=k_pool.stride(3),
        v_stride_0=v_pool.stride(0), v_stride_1=v_pool.stride(1),
        v_stride_2=v_pool.stride(2), v_stride_3=v_pool.stride(3),
        num_queries_per_kv=num_queries_per_kv, block_size=metadata.block_size,
        head_size=q.shape[-1], BLOCK_D=triton.next_power_of_2(q.shape[-1]), TILE=64,
    )
    return out


@triton.jit
def _paged_tree_gqa_fp32(
    out_ptr, q_ptr, k_ptr, v_ptr,
    prefix_lens_ptr, node_counts_ptr, cu_query_ptr,
    block_tables_ptr, tree_slots_ptr, bias_ptr, bias_offsets_ptr,
    scale,
    q_stride_0, q_stride_1, out_stride_0, out_stride_1, table_stride_0,
    k_stride_0, k_stride_1, k_stride_2, k_stride_3: tl.constexpr,
    v_stride_0, v_stride_1, v_stride_2, v_stride_3: tl.constexpr,
    GROUPS: tl.constexpr, GROUP_PAD: tl.constexpr, QUERY_TILE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr, TILE: tl.constexpr,
):
    query_block = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.program_id(2)
    context_len = tl.load(prefix_lens_ptr + request)
    node_count = tl.load(node_counts_ptr + request)
    tree_start = tl.load(cu_query_ptr + request)
    bias_start = tl.load(bias_offsets_ptr + request)
    # Programs never straddle request boundaries, including ragged tails.
    rows = tl.arange(0, QUERY_TILE * GROUP_PAD)
    local_q = query_block * QUERY_TILE + rows // GROUP_PAD
    group_head = rows % GROUP_PAD
    query_head = kv_head * GROUPS + group_head
    row_valid = (local_q < node_count) & (group_head < GROUPS)
    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < HEAD_SIZE
    q = tl.load(q_ptr + (tree_start + local_q[:, None]) * q_stride_0
                + query_head[:, None] * q_stride_1 + dims[None, :],
                mask=row_valid[:, None] & dim_valid[None, :], other=0.0).to(tl.float32)
    running_max = tl.full((QUERY_TILE * GROUP_PAD,), float("-inf"), tl.float32)
    running_sum = tl.zeros((QUERY_TILE * GROUP_PAD,), tl.float32)
    accumulator = tl.zeros((QUERY_TILE * GROUP_PAD, BLOCK_D), tl.float32)
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
        addresses = slots // BLOCK_SIZE
        offsets = slots % BLOCK_SIZE
        k = tl.load(k_ptr + addresses[:, None] * k_stride_0 + offsets[:, None] * k_stride_1
                    + kv_head * k_stride_2 + dims[None, :] * k_stride_3,
                    mask=valid[:, None] & dim_valid[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
        bias = tl.load(bias_ptr + bias_start + local_q[:, None] * node_count + node[None, :],
                       mask=row_valid[:, None] & valid[None, :] & in_tree[None, :], other=0.0).to(tl.float32)
        visible = row_valid[:, None] & valid[None, :] & (positions[None, :] <= context_len + local_q[:, None])
        scores = tl.where(visible, scores + bias, float("-inf"))
        tile_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, tile_max)
        next_max = tl.where(next_max > float("-inf"), next_max, 0.0)
        alpha = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max[:, None])
        v = tl.load(v_ptr + addresses[:, None] * v_stride_0 + offsets[:, None] * v_stride_1
                    + kv_head * v_stride_2 + dims[None, :] * v_stride_3,
                    mask=valid[:, None] & dim_valid[None, :], other=0.0).to(tl.float32)
        accumulator = accumulator * alpha[:, None] + tl.sum(probabilities[:, :, None] * v[None, :, :], axis=1)
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = next_max
    output = accumulator / tl.where(row_valid, running_sum, 1.0)[:, None]
    tl.store(out_ptr + (tree_start + local_q[:, None]) * out_stride_0
             + query_head[:, None] * out_stride_1 + dims[None, :], output,
             mask=row_valid[:, None] & dim_valid[None, :])


def packed_tree_attention_gqa(q, k_pool, v_pool, metadata, scale,
                              num_queries_per_kv, *, output_dtype=None,
                              query_tile=1, num_warps=4):
    """Reuse paged K/V across grouped heads and request-local query tiles.

    ``query_tile`` and ``num_warps`` are explicit kernel experiment parameters,
    not serving/tree-budget policies. The numerical tile remains exactly 64.
    """
    if isinstance(query_tile, bool) or query_tile not in (1, 2, 4):
        raise ValueError("query_tile must be 1, 2 or 4")
    if isinstance(num_warps, bool) or num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8 or 16")
    out = _validate(q, k_pool, v_pool, metadata, num_queries_per_kv, output_dtype)
    grid = (triton.cdiv(max(metadata.node_counts_host), query_tile), k_pool.shape[2], metadata.num_requests)
    _paged_tree_gqa_fp32[grid](
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
        GROUPS=num_queries_per_kv, GROUP_PAD=triton.next_power_of_2(num_queries_per_kv),
        QUERY_TILE=query_tile, BLOCK_SIZE=metadata.block_size, HEAD_SIZE=q.shape[-1],
        BLOCK_D=triton.next_power_of_2(q.shape[-1]), TILE=64, num_warps=num_warps,
    )
    return out
