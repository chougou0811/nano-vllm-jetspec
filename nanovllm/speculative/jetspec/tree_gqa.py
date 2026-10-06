"""Grouped Tensor Core tree attention, adapted from JetSpec 2c7b3fa.

One program handles several local tree rows AND all Q heads of one KV head.
Canonical prefix pages and transaction scratch slots are read directly; no KV
gather or global score matrix. The packed Q-block grid depends only on total Q
and request count, so graph replay can change individual ragged node counts.
The explicit legacy FP32-P backend remains available. This opt-in path uses the
official BF16-P/FP32-accumulator arithmetic, not legacy bitwise equivalence.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from nanovllm.speculative.jetspec.tree_attention import _validate


@triton.jit
def _request_for_q_block(cu_ptr, block, NUM_REQUESTS: tl.constexpr, Q_TILE: tl.constexpr):
    left, right = 0, NUM_REQUESTS
    while left < right:
        middle = (left + right) // 2
        start = tl.load(cu_ptr + middle) // Q_TILE + middle
        if start <= block:
            left = middle + 1
        else:
            right = middle
    return left - 1


@triton.jit
def _packed_tree_gqa_dot(
    out_ptr, q_ptr, k_ptr, v_ptr, prefix_ptr, count_ptr, cu_ptr,
    table_ptr, slots_ptr, bias_ptr, bias_offset_ptr, scale,
    q_s0, q_s1, out_s0, out_s1, table_s0,
    k_s0, k_s1, k_s2, k_s3: tl.constexpr,
    v_s0, v_s1, v_s2, v_s3: tl.constexpr,
    NUM_REQUESTS: tl.constexpr, GROUPS: tl.constexpr, Q_TILE: tl.constexpr,
    M: tl.constexpr, PAGE: tl.constexpr, D: tl.constexpr, PAD_D: tl.constexpr,
    TILE: tl.constexpr, PREFIX_SCALAR: tl.constexpr, PRUNE: tl.constexpr,
    FP32_P: tl.constexpr,
):
    block, kv_head = tl.program_id(0), tl.program_id(1)
    request = _request_for_q_block(cu_ptr, block, NUM_REQUESTS, Q_TILE)
    q_start = tl.load(cu_ptr + request)
    count = tl.load(count_ptr + request)
    local_block = block - (q_start // Q_TILE + request)
    if local_block * Q_TILE >= count:
        return
    prefix = tl.load(prefix_ptr + request)
    bias_start = tl.load(bias_offset_ptr + request)
    rows, dims, keys = tl.arange(0, M), tl.arange(0, PAD_D), tl.arange(0, TILE)
    local_q = local_block * Q_TILE + rows // GROUPS
    head = kv_head * GROUPS + rows % GROUPS
    row_ok = (rows < Q_TILE * GROUPS) & (local_q < count)
    dim_ok = dims < D
    q = tl.load(q_ptr + (q_start + local_q[:, None]) * q_s0
                + head[:, None] * q_s1 + dims[None, :],
                mask=row_ok[:, None] & dim_ok[None, :], other=0)
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denominator = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, PAD_D), tl.float32)
    # Validated parent-before-child ordering: no row in this block can see a
    # later local tree node. Ancestor bias still removes sibling/off-path keys.
    end = prefix + count
    if PRUNE:
        end = prefix + tl.minimum(count, (local_block + 1) * Q_TILE)
    for tile in range(tl.cdiv(end, TILE)):
        pos = tile * TILE + keys
        valid = pos < end
        is_prefix = pos < prefix
        node = pos - prefix
        in_tree = (node >= 0) & (node < count)
        if PREFIX_SCALAR and PAGE % TILE == 0 and (tile + 1) * TILE <= prefix:
            scalar_page = tl.load(table_ptr + request * table_s0 + (tile * TILE) // PAGE).to(tl.int64)
            blocks = scalar_page + tl.zeros((TILE,), tl.int64)
            offsets = (pos % PAGE).to(tl.int64)
        else:
            page = tl.load(table_ptr + request * table_s0 + pos // PAGE,
                           mask=valid & is_prefix, other=0).to(tl.int64)
            scratch = tl.load(slots_ptr + q_start + node,
                              mask=valid & in_tree, other=0).to(tl.int64)
            slot = tl.where(is_prefix, page * PAGE + pos % PAGE, scratch)
            blocks, offsets = slot // PAGE, slot % PAGE
        # One K/V tile per KV head, reused across Q_TILE * GROUPS score rows.
        k = tl.load(k_ptr + blocks[None, :] * k_s0 + offsets[None, :] * k_s1
                    + kv_head * k_s2 + dims[:, None] * k_s3,
                    mask=dim_ok[:, None] & valid[None, :], other=0)
        scores = tl.dot(q, k, out_dtype=tl.float32) * scale
        bias = tl.load(bias_ptr + bias_start + local_q[:, None] * count + node[None, :],
                       mask=row_ok[:, None] & valid[None, :] & in_tree[None, :], other=0)
        visible = row_ok[:, None] & valid[None, :] & (pos[None, :] <= prefix + local_q[:, None])
        scores = tl.where(visible, scores + bias, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        next_max = tl.where(next_max > -float("inf"), next_max, 0.)
        probability = tl.exp(scores - next_max[:, None])
        alpha = tl.exp(maximum - next_max)
        v = tl.load(v_ptr + blocks[:, None] * v_s0 + offsets[:, None] * v_s1
                    + kv_head * v_s2 + dims[None, :] * v_s3,
                    mask=valid[:, None] & dim_ok[None, :], other=0)
        if FP32_P:
            pv = tl.dot(probability, v.to(tl.float32), input_precision="tf32x3")
        else:
            pv = tl.dot(probability.to(v.dtype), v, out_dtype=tl.float32)
        acc = acc * alpha[:, None] + pv
        denominator = denominator * alpha + tl.sum(probability, 1)
        maximum = next_max
    result = acc / tl.where(row_ok, denominator, 1.)[:, None]
    tl.store(out_ptr + (q_start + local_q[:, None]) * out_s0 + head[:, None] * out_s1
             + dims[None, :], result, mask=row_ok[:, None] & dim_ok[None, :])


def packed_tree_attention_gqa(q, k_pool, v_pool, metadata, scale, num_queries_per_kv,
                              *, output_dtype=None, query_tile=4, prefix_scalar=True,
                              prune=True, fp32_probability=False):
    """One genuine packed launch; switches are operator-only ablation controls.

    Serving fixes Q_TILE=4/TILE64/4 warps, prefix addressing and pruning on.
    Ragged grouping uses device cu offsets, never capture-time host max(counts).
    Tensor Core padded lanes are masked and never cross request ownership.
    """
    if query_tile not in (1, 4) or isinstance(query_tile, bool):
        raise ValueError("GQA query tile must be 1 or 4")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("grouped Tensor Core tree attention requires BF16 or FP16")
    if num_queries_per_kv not in (1, 2, 4, 8, 16):
        raise ValueError("grouped tree attention requires power-of-two GQA <=16")
    out = _validate(q, k_pool, v_pool, metadata, num_queries_per_kv, output_dtype)
    _packed_tree_gqa_dot[(metadata.total_queries // query_tile + metadata.num_requests, k_pool.shape[2])](
        out, q, k_pool, v_pool, metadata.prefix_lens, metadata.node_counts,
        metadata.cu_seqlens_q, metadata.block_tables, metadata.tree_slots,
        metadata.qq_bias, metadata.qq_bias_offsets, scale,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1), metadata.block_tables.stride(0),
        *k_pool.stride(), *v_pool.stride(), NUM_REQUESTS=metadata.num_requests,
        GROUPS=num_queries_per_kv, Q_TILE=query_tile,
        M=max(16, query_tile * num_queries_per_kv), PAGE=metadata.block_size,
        D=q.shape[-1], PAD_D=max(32, triton.next_power_of_2(q.shape[-1])), TILE=64,
        PREFIX_SCALAR=prefix_scalar, PRUNE=prune, FP32_P=fp32_probability, num_warps=4)
    return out
