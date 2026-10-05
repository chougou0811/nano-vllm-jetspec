"""JetSpec paged-tree attention with an explicit FP32 numerical contract.

Both legacy c1 and genuinely packed ragged verification use FP32 multiply,
reduction and online softmax with TILE=64. No batch-size-triggered switch to
the upstream BF16 probability-matmul implementation is permitted.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata


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


@triton.jit
def _packed_paged_tree_fp32(
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
    """One (packed query, query head) program, with request-local addressing.

    The arithmetic below intentionally matches the qualified c1 specialization.
    Only address resolution differs: canonical prefix page table + tree scratch
    subrange, rather than a persistent history-sized logical slot vector.
    """
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


@lru_cache(maxsize=None)
def _tree_attention_device_capability(device_index: int) -> tuple[int, int]:
    """Resolve hardware once, not through a device query in every layer."""
    return torch.cuda.get_device_capability(device_index)


def _use_prefix_tree_attention(q, k_pool, v_pool, metadata, groups) -> bool:
    """Measured SM120/Qwen3 geometry only; unsupported devices stay frozen.

    Host metadata selects the path without .item()/tolist() or GPU barriers.
    c1 short-prefix launch overhead did not amortize in the preregistered
    operator matrix. Multi-request tiles amortize from one complete TILE64.
    This changes neither tree policy nor request scheduling.
    """
    if (q.dtype != torch.bfloat16 or k_pool.dtype != q.dtype or v_pool.dtype != q.dtype
            or q.shape[1:] != (32, 128) or k_pool.shape[2:] != (8, 128)
            or groups != 4 or metadata.block_size != 256):
        return False
    longest_prefix = max(metadata.prefix_lengths, default=0)
    if longest_prefix < (256 if metadata.num_requests == 1 else 64):
        return False
    return _tree_attention_device_capability(q.device.index) == (12, 0)


def packed_tree_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    metadata: PackedTreeMetadata,
    scale: float,
    num_queries_per_kv: int,
    *,
    backend: str = "auto",
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """One ragged launch with independent request/ancestor visibility.

    ``auto`` uses the qualified address-only prefix specialization on measured
    SM120/BF16/Qwen3 geometry, otherwise the frozen scalar FP32 kernel.
    Explicit reference/prefix entries support unchanged native operands and
    FP32 output stores for numerical diagnostics, never BF16 probabilities.
    """
    if backend not in ("auto", "reference", "prefix"):
        raise ValueError("tree attention backend must be auto, reference or prefix")
    if q.ndim != 3 or q.shape[0] != metadata.total_queries:
        raise ValueError("packed queries do not match the metadata")
    if k_pool.ndim != 4 or k_pool.shape != v_pool.shape or k_pool.shape[1] != metadata.block_size:
        raise ValueError("packed K/V page geometry does not match metadata")
    if not q.is_cuda or q.device != k_pool.device or q.device != v_pool.device or q.device != metadata.tree_slots.device:
        raise ValueError("packed attention requires CUDA queries, KV and metadata on one device")
    total_q, num_query_heads, head_size = q.shape
    if q.stride(-1) != 1 or k_pool.shape[3] != head_size:
        raise ValueError("unsupported packed query/KV head geometry")
    if num_queries_per_kv <= 0 or num_query_heads != k_pool.shape[2] * num_queries_per_kv:
        raise ValueError("invalid packed GQA head grouping")
    if output_dtype not in (None, q.dtype, torch.float32):
        raise ValueError("output dtype must match input or be FP32 for qualification")
    if backend == "reference":
        from nanovllm.speculative.jetspec.tree_attention import packed_tree_attention_reference
        return packed_tree_attention_reference(q, k_pool, v_pool, metadata, scale,
            num_queries_per_kv, output_dtype=output_dtype)
    if backend == "prefix":
        from nanovllm.speculative.jetspec.tree_prefix import packed_tree_attention_prefix_split_exact
        return packed_tree_attention_prefix_split_exact(q, k_pool, v_pool, metadata, scale,
            num_queries_per_kv, output_dtype=output_dtype, num_warps=4)
    out = torch.empty_like(q) if output_dtype is None else torch.empty_like(q, dtype=output_dtype)
    if _use_prefix_tree_attention(q, k_pool, v_pool, metadata, num_queries_per_kv):
        from nanovllm.speculative.jetspec.tree_prefix import packed_tree_attention_prefix_prevalidated
        return packed_tree_attention_prefix_prevalidated(q, k_pool, v_pool, metadata, scale,
            num_queries_per_kv, out, num_warps=4)
    _packed_paged_tree_fp32[(total_q, num_query_heads)](
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
        head_size=head_size, BLOCK_D=triton.next_power_of_2(head_size), TILE=64,
    )
    return out


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
    if num_seqs != 1:
        raise ValueError("multi-request tree verification requires explicit PackedTreeMetadata")
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
