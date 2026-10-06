"""HF-order RoPE and provisional KV scatter in one kernel.

The input Q/K have ALREADY passed reference Q/K RMSNorm. This operation changes
neither GEMM shape nor normalization/reduction arithmetic. In particular, every
BF16 product is rounded before its add/subtract, exactly as eager HF RoPE does;
FP32 multiplication followed by one final BF16 cast would be a different op.

``rope_scatter`` validates arbitrary external operands, including device-side
addresses, and therefore synchronizes. The runner's checked transaction can use
``rope_scatter_prevalidated`` after reserving its unique scratch slots; that
boundary performs only the launch and never allocates, downloads or syncs.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate_halves(x1, x2, cos, sin, DTYPE: tl.constexpr):
    # Keep all four eager BF16 multiply rounding boundaries. Disabling FMA at
    # launch is also required for the FP32 case and for the add/subtract below.
    a = (x1 * cos).to(DTYPE).to(tl.float32)
    b = (x2 * sin).to(DTYPE).to(tl.float32)
    c = (x2 * cos).to(DTYPE).to(tl.float32)
    d = (x1 * sin).to(DTYPE).to(tl.float32)
    return (a - b).to(DTYPE), (c + d).to(DTYPE)


@triton.jit
def _rope_scatter_hf_order(
    q_ptr, k_ptr, v_ptr, positions_ptr, cache_ptr, slots_ptr,
    out_ptr, k_pool_ptr, v_pool_ptr,
    q_s0, q_s1, q_s2, k_s0, k_s1, k_s2, v_s0, v_s1, v_s2,
    positions_stride, slots_stride, cache_s0, cache_slast,
    out_s0, out_s1, out_s2,
    kp_s0, kp_s1, kp_s2, kp_s3, vp_s0, vp_s1, vp_s2, vp_s3,
    KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr, HALF_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    half = HEAD_DIM // 2
    dims = tl.arange(0, HALF_BLOCK)
    valid = dims < half
    position = tl.load(positions_ptr + row * positions_stride).to(tl.int64)
    cache_base = cache_ptr + position * cache_s0
    dtype = q_ptr.dtype.element_ty
    cos = tl.load(cache_base + dims * cache_slast, mask=valid,
                  other=0).to(dtype).to(tl.float32)
    sin = tl.load(cache_base + (dims + half) * cache_slast, mask=valid,
                  other=0).to(dtype).to(tl.float32)
    q_base = q_ptr + row * q_s0 + head * q_s1
    q1 = tl.load(q_base + dims * q_s2, mask=valid, other=0).to(tl.float32)
    q2 = tl.load(q_base + (dims + half) * q_s2, mask=valid,
                 other=0).to(tl.float32)
    rotated_q1, rotated_q2 = _rotate_halves(q1, q2, cos, sin, dtype)
    out_base = out_ptr + row * out_s0 + head * out_s1
    tl.store(out_base + dims * out_s2, rotated_q1, mask=valid)
    tl.store(out_base + (dims + half) * out_s2, rotated_q2, mask=valid)

    # Every KV head has exactly ONE writer. GQA's remaining query-head programs
    # produce Q only, rather than redundantly racing on identical K/V stores.
    if head < KV_HEADS:
        slot = tl.load(slots_ptr + row * slots_stride).to(tl.int64)
        page = slot // PAGE_SIZE
        offset = slot % PAGE_SIZE
        k_base = k_ptr + row * k_s0 + head * k_s1
        k1 = tl.load(k_base + dims * k_s2, mask=valid, other=0).to(tl.float32)
        k2 = tl.load(k_base + (dims + half) * k_s2, mask=valid,
                     other=0).to(tl.float32)
        rotated_k1, rotated_k2 = _rotate_halves(k1, k2, cos, sin, dtype)
        kp_base = k_pool_ptr + page * kp_s0 + offset * kp_s1 + head * kp_s2
        tl.store(kp_base + dims * kp_s3, rotated_k1, mask=valid)
        tl.store(kp_base + (dims + half) * kp_s3, rotated_k2, mask=valid)
        v_base = v_ptr + row * v_s0 + head * v_s1
        value1 = tl.load(v_base + dims * v_s2, mask=valid, other=0)
        value2 = tl.load(v_base + (dims + half) * v_s2, mask=valid, other=0)
        vp_base = v_pool_ptr + page * vp_s0 + offset * vp_s1 + head * vp_s2
        tl.store(vp_base + dims * vp_s3, value1, mask=valid)
        tl.store(vp_base + (dims + half) * vp_s3, value2, mask=valid)


def _memory_range(tensor):
    # All accepted strides are positive. Bounds conservatively include holes in
    # noncontiguous views; no device data is materialized to check aliasing.
    start = tensor.data_ptr()
    span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return start, start + span * tensor.element_size()


def _overlap(a, b):
    a0, a1 = _memory_range(a)
    b0, b1 = _memory_range(b)
    return a0 < b1 and b0 < a1


def _nonoverlapping(tensor):
    # A conservative host-only injectivity check. Ordinary dense, permuted and
    # sliced layouts pass; arbitrary overlapping as_strided write views do not.
    span = 1
    for stride, size in sorted((stride, size) for stride, size in
                               zip(tensor.stride(), tensor.shape) if size > 1):
        if stride < span:
            return False
        span += (size - 1) * stride
    return True


def _validate(q, k, v, positions, cache, k_pool, v_pool, node_slots, out):
    tensors = (q, k, v, positions, cache, k_pool, v_pool, node_slots)
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Q/K/V must be [rows,heads,dim] with equal K/V geometry")
    rows, q_heads, dim = q.shape
    if (rows < 1 or q_heads < 1 or k.shape[0] != rows or k.shape[1] < 1 or
            k.shape[2] != dim or dim < 2 or dim % 2 or dim > 256 or
            q_heads % k.shape[1]):
        raise ValueError("invalid nonempty Q/K/V GQA or even head geometry")
    if q.dtype not in (torch.bfloat16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("RoPE fusion supports matching BF16 or FP32 Q/K/V")
    if (positions.shape != (rows,) or node_slots.shape != (rows,) or
            positions.dtype not in (torch.int32, torch.int64) or
            node_slots.dtype not in (torch.int32, torch.int64)):
        raise ValueError("positions and node_slots must be integer [rows] vectors")
    if (cache.ndim not in (2, 3) or cache.shape[-1] != dim or cache.shape[0] < 1 or
            (cache.ndim == 3 and cache.shape[1] != 1) or
            cache.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("cos/sin cache must be BF16/FP32 [positions,dim] or [positions,1,dim]")
    if (k_pool.ndim != 4 or v_pool.shape != k_pool.shape or
            k_pool.shape[0] < 1 or k_pool.shape[1] < 1 or
            tuple(k_pool.shape[2:]) != (k.shape[1], dim) or
            k_pool.dtype != q.dtype or v_pool.dtype != q.dtype):
        raise ValueError("KV pools must match [pages,page_size,kv_heads,dim] and Q/K/V dtype")
    if any(any(stride <= 0 for stride in tensor.stride()) for tensor in tensors):
        raise ValueError("RoPE fusion requires positive tensor strides")
    if not q.is_cuda or any(tensor.device != q.device for tensor in tensors):
        raise ValueError("RoPE fusion requires CUDA tensors on one device")
    if not _nonoverlapping(k_pool) or not _nonoverlapping(v_pool):
        raise ValueError("KV pool layouts must not have overlapping write addresses")
    if out is not None:
        if (out.shape != q.shape or out.dtype != q.dtype or out.device != q.device or
                any(stride <= 0 for stride in out.stride())):
            raise ValueError("output must match Q shape, dtype and device with positive strides")
        if not _nonoverlapping(out):
            raise ValueError("output layout must not have overlapping write addresses")
        # A write can clobber another head's source when arbitrary strided views
        # overlap. Qualified production outputs are separate allocations.
        if any(_overlap(out, tensor) for tensor in tensors):
            raise ValueError("output must not alias any source or KV pool")
    if _overlap(k_pool, v_pool):
        raise ValueError("K and V pool views must not overlap")
    sources = (q, k, v, positions, cache, node_slots)
    if any(_overlap(pool, source) for pool in (k_pool, v_pool) for source in sources):
        raise ValueError("KV pools must not overlap source tensors")
    # Safe arbitrary-input boundary. These deliberate checks synchronize and
    # must NOT be repeated inside the serving hot path or CUDA-graph capture.
    if bool(((positions < 0) | (positions >= cache.shape[0])).any().item()):
        raise ValueError("RoPE positions lie outside the cos/sin cache")
    capacity = k_pool.shape[0] * k_pool.shape[1]
    if bool(((node_slots < 0) | (node_slots >= capacity)).any().item()):
        raise ValueError("node slots lie outside the KV pool")
    if node_slots.unique().numel() != rows:
        raise ValueError("node slots must be unique to avoid KV write races")


def rope_scatter(q, k, v, positions, cos_sin_cache, k_pool, v_pool, node_slots, *, out=None):
    """Validate and return rotated Q, scattering rotated K and raw V in place.

    Cache rows contain ``[cos_half | sin_half]``. Q/K/V and addresses may be
    noncontiguous; KV destination slots must be in-range and unique. This
    diagnostic/external wrapper deliberately checks device address VALUES.
    """
    _validate(q, k, v, positions, cos_sin_cache, k_pool, v_pool, node_slots, out)
    if out is None:
        out = torch.empty_like(q)
    return rope_scatter_prevalidated(q, k, v, positions, cos_sin_cache,
                                    k_pool, v_pool, node_slots, out)


def rope_scatter_prevalidated(q, k, v, positions, cos_sin_cache, k_pool, v_pool, node_slots, out):
    """Launch-only runner boundary; see ``rope_scatter`` for required invariants.

    Caller guarantees valid/unique reserved scratch slots, in-range RoPE
    positions, BF16/FP32 matching dtypes, disjoint output/source storage and
    supported geometry. Writes are provisional; acceptance/commit stays owned
    by the existing BatchTreeTransaction. No metadata ownership is transferred.
    """
    _rope_scatter_hf_order[(q.shape[0], q.shape[1])](
        q, k, v, positions, cos_sin_cache, node_slots, out, k_pool, v_pool,
        *q.stride(), *k.stride(), *v.stride(),
        positions.stride(0), node_slots.stride(0),
        cos_sin_cache.stride(0), cos_sin_cache.stride(-1),
        *out.stride(), *k_pool.stride(), *v_pool.stride(),
        KV_HEADS=k.shape[1], HEAD_DIM=q.shape[-1], PAGE_SIZE=k_pool.shape[1],
        HALF_BLOCK=triton.next_power_of_2(q.shape[-1] // 2),
        num_warps=4, enable_fp_fusion=False,
    )
    return out
