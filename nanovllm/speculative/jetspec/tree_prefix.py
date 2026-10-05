"""Scalar-query Tree Attention with a page-contiguous prefix fast path.

Complete prefix tiles require neither per-key tree-slot resolution nor ancestor
bias/visibility loads. Page sizes divisible by the qualified TILE=64 guarantee
that such a tile lives in one page, so it resolves one scalar page-table entry.
The mixed prefix tail and tree retain the frozen reference addressing/masking.
FP32 QK/PV reductions, online softmax, and key tile order remain unchanged.

This is an explicit experimental operator, not a production serving dispatch.
"""
from __future__ import annotations

import triton
import triton.language as tl

from nanovllm.speculative.jetspec.tree_attention import (
    _validate,
    packed_tree_attention_reference,
)


@triton.jit
def _packed_tree_prefix_fp32(
    out_ptr, q_ptr, k_ptr, v_ptr,
    query_request_ptr, query_local_ptr, prefix_lens_ptr, node_counts_ptr,
    cu_query_ptr, block_tables_ptr, tree_slots_ptr, bias_ptr, bias_offsets_ptr,
    scale,
    q_stride_0, q_stride_1, out_stride_0, out_stride_1, table_stride_0,
    k_stride_0, k_stride_1, k_stride_2, k_stride_3: tl.constexpr,
    v_stride_0, v_stride_1, v_stride_2, v_stride_3: tl.constexpr,
    GROUPS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr, BLOCK_D: tl.constexpr, TILE: tl.constexpr,
):
    query_idx = tl.program_id(0)
    query_head = tl.program_id(1)
    request = tl.load(query_request_ptr + query_idx)
    local_query = tl.load(query_local_ptr + query_idx)
    context_len = tl.load(prefix_lens_ptr + request)
    query_len = tl.load(node_counts_ptr + request)
    tree_start = tl.load(cu_query_ptr + request)
    bias_start = tl.load(bias_offsets_ptr + request)
    seq_len = context_len + query_len
    kv_head = query_head // GROUPS
    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < HEAD_SIZE
    keys = tl.arange(0, TILE)
    q = tl.load(q_ptr + query_idx * q_stride_0 + query_head * q_stride_1 + dims,
                mask=dim_valid, other=0.0).to(tl.float32)
    running_max = tl.full((), float("-inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    prefix_tiles = context_len // TILE

    # TILE divides BLOCK_SIZE: a complete prefix tile cannot cross a page.
    # Every row is visible, so no tree addressing/bias loads are required.
    for tile_idx in range(0, prefix_tiles):
        tile_start = tile_idx * TILE
        page = tl.load(block_tables_ptr + request * table_stride_0
                       + tile_start // BLOCK_SIZE).to(tl.int64)
        page_offsets = tile_start % BLOCK_SIZE + keys
        k = tl.load(k_ptr + page * k_stride_0
                    + page_offsets[:, None] * k_stride_1 + kv_head * k_stride_2
                    + dims[None, :] * k_stride_3,
                    mask=dim_valid[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        # Same TILE=64 FP32 update and reduction axes as the frozen reference.
        tile_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, tile_max)
        next_max = tl.where(next_max > float("-inf"), next_max, 0.0)
        alpha = tl.exp(running_max - next_max)
        probability = tl.exp(scores - next_max)
        v = tl.load(v_ptr + page * v_stride_0
                    + page_offsets[:, None] * v_stride_1 + kv_head * v_stride_2
                    + dims[None, :] * v_stride_3,
                    mask=dim_valid[None, :], other=0.0).to(tl.float32)
        accumulator = accumulator * alpha + tl.sum(probability[:, None] * v, axis=0)
        running_sum = running_sum * alpha + tl.sum(probability, axis=0)
        running_max = next_max

    # Start at floor(P/TILE)*TILE, not P: retain the original reduction grouping
    # for the potentially mixed prefix/tree tile and every subsequent tree tile.
    for tile_idx in range(prefix_tiles, tl.cdiv(seq_len, TILE)):
        key_pos = tile_idx * TILE + keys
        valid = key_pos < seq_len
        is_prefix = key_pos < context_len
        key_rel = key_pos - context_len
        is_node = (key_rel >= 0) & (key_rel < query_len)
        prefix_block = tl.load(
            block_tables_ptr + request * table_stride_0 + key_pos // BLOCK_SIZE,
            mask=valid & is_prefix, other=0).to(tl.int64)
        tree_slot = tl.load(tree_slots_ptr + tree_start + key_rel,
                            mask=valid & is_node, other=0).to(tl.int64)
        slot = tl.where(is_prefix, prefix_block * BLOCK_SIZE + key_pos % BLOCK_SIZE,
                        tree_slot)
        physical_block = slot // BLOCK_SIZE
        physical_offset = slot % BLOCK_SIZE
        k = tl.load(k_ptr + physical_block[:, None] * k_stride_0
                    + physical_offset[:, None] * k_stride_1 + kv_head * k_stride_2
                    + dims[None, :] * k_stride_3,
                    mask=valid[:, None] & dim_valid[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        bias = tl.load(bias_ptr + bias_start + local_query * query_len + key_rel,
                       mask=valid & is_node, other=0.0).to(tl.float32)
        visible = valid & (key_pos <= context_len + local_query)
        scores = tl.where(visible, scores + bias, float("-inf"))
        tile_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, tile_max)
        next_max = tl.where(next_max > float("-inf"), next_max, 0.0)
        alpha = tl.exp(running_max - next_max)
        probability = tl.exp(scores - next_max)
        v = tl.load(v_ptr + physical_block[:, None] * v_stride_0
                    + physical_offset[:, None] * v_stride_1 + kv_head * v_stride_2
                    + dims[None, :] * v_stride_3,
                    mask=valid[:, None] & dim_valid[None, :], other=0.0).to(tl.float32)
        accumulator = accumulator * alpha + tl.sum(probability[:, None] * v, axis=0)
        running_sum = running_sum * alpha + tl.sum(probability, axis=0)
        running_max = next_max
    output = accumulator / running_sum
    tl.store(out_ptr + query_idx * out_stride_0 + query_head * out_stride_1 + dims,
             output, mask=dim_valid)


def packed_tree_attention_prefix(q, k_pool, v_pool, metadata, scale,
                                 num_queries_per_kv, *, output_dtype=None,
                                 num_warps=4):
    """Prefix-only address specialization with the original scalar-query grid.

    Page sizes not divisible by 64 explicitly use the reference operator;
    ``num_warps`` applies only to the prefix specialization, not that fallback.
    Neither serving/tree policy nor arithmetic precision is changed here.
    """
    if isinstance(num_warps, bool) or num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8 or 16")
    if metadata.block_size % 64:
        return packed_tree_attention_reference(q, k_pool, v_pool, metadata, scale,
                                               num_queries_per_kv,
                                               output_dtype=output_dtype)
    out = _validate(q, k_pool, v_pool, metadata, num_queries_per_kv, output_dtype)
    _packed_tree_prefix_fp32[(metadata.total_queries, q.shape[1])](
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
        GROUPS=num_queries_per_kv, BLOCK_SIZE=metadata.block_size,
        HEAD_SIZE=q.shape[-1], BLOCK_D=triton.next_power_of_2(q.shape[-1]),
        TILE=64, num_warps=num_warps,
    )
    return out


@triton.jit
def _packed_tree_prefix_single_loop_fp32(
    out_ptr, q_ptr, k_ptr, v_ptr,
    query_request_ptr, query_local_ptr, prefix_lens_ptr, node_counts_ptr,
    cu_query_ptr, block_tables_ptr, tree_slots_ptr, bias_ptr, bias_offsets_ptr,
    scale,
    q_stride_0, q_stride_1, out_stride_0, out_stride_1, table_stride_0,
    k_stride_0, k_stride_1, k_stride_2, k_stride_3: tl.constexpr,
    v_stride_0, v_stride_1, v_stride_2, v_stride_3: tl.constexpr,
    num_queries_per_kv: tl.constexpr, block_size: tl.constexpr,
    head_size: tl.constexpr, BLOCK_D: tl.constexpr, TILE: tl.constexpr,
):
    """Address-only experiment: one original loop and one FP32 update body."""
    query_idx = tl.program_id(0)
    query_head = tl.program_id(1)
    request = tl.load(query_request_ptr + query_idx)
    local_query = tl.load(query_local_ptr + query_idx)
    context_len = tl.load(prefix_lens_ptr + request)
    query_len = tl.load(node_counts_ptr + request)
    tree_start = tl.load(cu_query_ptr + request)
    bias_start = tl.load(bias_offsets_ptr + request)
    seq_len = context_len + query_len
    kv_head = query_head // num_queries_per_kv
    d = tl.arange(0, BLOCK_D)
    dmask = d < head_size
    q = tl.load(
        q_ptr + query_idx * q_stride_0 + query_head * q_stride_1 + d,
        mask=dmask,
        other=0.0,
    ).to(tl.float32)
    running_max = tl.full((), float("-inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    num_tiles = tl.cdiv(seq_len, TILE)
    for tile_idx in range(0, num_tiles):
        key_pos = tile_idx * TILE + tl.arange(0, TILE)
        valid = key_pos < seq_len
        is_prefix = key_pos < context_len
        key_rel = key_pos - context_len
        is_node = (key_rel >= 0) & (key_rel < query_len)
        if (tile_idx + 1) * TILE <= context_len:
            # The only altered part of the loop: TILE divides block_size, so
            # a complete prefix tile uses one page-table scalar, not a gather.
            prefix_page = tl.load(
                block_tables_ptr + request * table_stride_0
                + (tile_idx * TILE) // block_size,
            ).to(tl.int64)
            physical_block = prefix_page + tl.zeros((TILE,), tl.int64)
            physical_offset = (key_pos % block_size).to(tl.int64)
        else:
            prefix_block = tl.load(
                block_tables_ptr + request * table_stride_0 + key_pos // block_size,
                mask=valid & is_prefix,
                other=0,
            ).to(tl.int64)
            tree_slot = tl.load(
                tree_slots_ptr + tree_start + key_rel,
                mask=valid & is_node,
                other=0,
            ).to(tl.int64)
            slot = tl.where(is_prefix, prefix_block * block_size + key_pos % block_size, tree_slot)
            physical_block = slot // block_size
            physical_offset = slot % block_size
        # Below is deliberately the frozen scalar kernel's one update body,
        # including prefix bias-zero/visibility expressions and FP32 order.
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
        bias = tl.load(
            bias_ptr + bias_start + local_query * query_len + key_rel,
            mask=valid & is_node,
            other=0.0,
        ).to(tl.float32)
        visible = valid & (key_pos <= context_len + local_query)
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


def packed_tree_attention_prefix_single_loop(q, k_pool, v_pool, metadata, scale,
                                             num_queries_per_kv, *, output_dtype=None,
                                             num_warps=4):
    """Optional address-only variant retaining the original single tile loop.

    Default scalar grid/warps/TILE64 and one shared arithmetic body match the
    frozen reference. This does not promise compiled bitwise equivalence and
    requires same-state trained qualification; no default dispatch is changed.
    Unsupported page geometry explicitly returns the reference operator.
    """
    if isinstance(num_warps, bool) or num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8 or 16")
    if metadata.block_size % 64:
        return packed_tree_attention_reference(q, k_pool, v_pool, metadata, scale,
                                               num_queries_per_kv,
                                               output_dtype=output_dtype)
    out = _validate(q, k_pool, v_pool, metadata, num_queries_per_kv, output_dtype)
    _packed_tree_prefix_single_loop_fp32[(metadata.total_queries, q.shape[1])](
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
        head_size=q.shape[-1], BLOCK_D=triton.next_power_of_2(q.shape[-1]),
        TILE=64, num_warps=num_warps,
    )
    return out


@triton.jit
def _packed_tree_prefix_split_exact_fp32(
    out_ptr, q_ptr, k_ptr, v_ptr,
    query_request_ptr, query_local_ptr, prefix_lens_ptr, node_counts_ptr,
    cu_query_ptr, block_tables_ptr, tree_slots_ptr, bias_ptr, bias_offsets_ptr,
    scale,
    q_stride_0, q_stride_1, out_stride_0, out_stride_1, table_stride_0,
    k_stride_0, k_stride_1, k_stride_2, k_stride_3: tl.constexpr,
    v_stride_0, v_stride_1, v_stride_2, v_stride_3: tl.constexpr,
    num_queries_per_kv: tl.constexpr, block_size: tl.constexpr,
    head_size: tl.constexpr, BLOCK_D: tl.constexpr, TILE: tl.constexpr,
):
    """Two loops, each retaining every original scalar-kernel math expression."""
    query_idx = tl.program_id(0)
    query_head = tl.program_id(1)
    request = tl.load(query_request_ptr + query_idx)
    local_query = tl.load(query_local_ptr + query_idx)
    context_len = tl.load(prefix_lens_ptr + request)
    query_len = tl.load(node_counts_ptr + request)
    tree_start = tl.load(cu_query_ptr + request)
    bias_start = tl.load(bias_offsets_ptr + request)
    seq_len = context_len + query_len
    kv_head = query_head // num_queries_per_kv
    d = tl.arange(0, BLOCK_D)
    dmask = d < head_size
    q = tl.load(
        q_ptr + query_idx * q_stride_0 + query_head * q_stride_1 + d,
        mask=dmask,
        other=0.0,
    ).to(tl.float32)
    running_max = tl.full((), float("-inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    num_tiles = tl.cdiv(seq_len, TILE)
    prefix_tiles = context_len // TILE
    for tile_idx in range(0, prefix_tiles):
        key_pos = tile_idx * TILE + tl.arange(0, TILE)
        valid = key_pos < seq_len
        key_rel = key_pos - context_len
        is_node = (key_rel >= 0) & (key_rel < query_len)
        prefix_page = tl.load(
            block_tables_ptr + request * table_stride_0
            + (tile_idx * TILE) // block_size,
        ).to(tl.int64)
        physical_block = prefix_page + tl.zeros((TILE,), tl.int64)
        physical_offset = (key_pos % block_size).to(tl.int64)
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
        bias = tl.load(
            bias_ptr + bias_start + local_query * query_len + key_rel,
            mask=valid & is_node,
            other=0.0,
        ).to(tl.float32)
        visible = valid & (key_pos <= context_len + local_query)
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
    for tile_idx in range(prefix_tiles, num_tiles):
        key_pos = tile_idx * TILE + tl.arange(0, TILE)
        valid = key_pos < seq_len
        is_prefix = key_pos < context_len
        key_rel = key_pos - context_len
        is_node = (key_rel >= 0) & (key_rel < query_len)
        prefix_block = tl.load(
            block_tables_ptr + request * table_stride_0 + key_pos // block_size,
            mask=valid & is_prefix,
            other=0,
        ).to(tl.int64)
        tree_slot = tl.load(
            tree_slots_ptr + tree_start + key_rel,
            mask=valid & is_node,
            other=0,
        ).to(tl.int64)
        slot = tl.where(is_prefix, prefix_block * block_size + key_pos % block_size, tree_slot)
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
        bias = tl.load(
            bias_ptr + bias_start + local_query * query_len + key_rel,
            mask=valid & is_node,
            other=0.0,
        ).to(tl.float32)
        visible = valid & (key_pos <= context_len + local_query)
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


def packed_tree_attention_prefix_split_exact(q, k_pool, v_pool, metadata, scale,
                                            num_queries_per_kv, *, output_dtype=None,
                                            num_warps=4):
    """Optional two-loop experiment with exact source-level reference math.

    Prefix tiles retain original masked bias loads, visibility, K/V masks and
    every FP32 expression; only address resolution is specialized. The suffix
    starts on the original floor(P/64)*64 boundary. The name describes retained
    source expressions, NOT a promise of compiled bitwise equality or passing
    the trained numerical envelope. Existing/default operators are untouched.
    """
    if isinstance(num_warps, bool) or num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8 or 16")
    if metadata.block_size % 64:
        return packed_tree_attention_reference(q, k_pool, v_pool, metadata, scale,
                                               num_queries_per_kv,
                                               output_dtype=output_dtype)
    out = _validate(q, k_pool, v_pool, metadata, num_queries_per_kv, output_dtype)
    return packed_tree_attention_prefix_prevalidated(q, k_pool, v_pool, metadata,
        scale, num_queries_per_kv, out, num_warps=num_warps)


def packed_tree_attention_prefix_prevalidated(q, k_pool, v_pool, metadata, scale,
                                             num_queries_per_kv, out, *, num_warps=4):
    """Internal serving launch after dispatcher validation/output allocation.

    The checked standalone entry above calls this SAME JIT launch. Avoid
    repeated Q/K/V/device checks and a second output preparation in every
    Target layer. Callers must already establish CUDA/page/head geometry;
    this is not an unchecked alternative public attention backend.
    """
    _packed_tree_prefix_split_exact_fp32[(metadata.total_queries, q.shape[1])](
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
        head_size=q.shape[-1], BLOCK_D=triton.next_power_of_2(q.shape[-1]),
        TILE=64, num_warps=num_warps,
    )
    return out
