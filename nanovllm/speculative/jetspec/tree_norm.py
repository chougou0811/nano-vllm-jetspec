"""Explicit reference-order RMSNorm fusion experiment for Target verification.

This module is not an automatic serving backend. It preserves the reference
FP32 square/mean/rsqrt and, importantly, the input-dtype rounding *before* the
weight multiply. Triton's reduction order can differ from eager PyTorch, so
matching source expressions is not a bitwise or trained-model qualification
claim. Integration must pass the existing, unchanged numerical gates first.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _reference_order_rms_norm(
    out_ptr, x_ptr, weight_ptr, eps,
    x_stride_row: tl.constexpr, x_stride_head: tl.constexpr,
    x_stride_dim: tl.constexpr, weight_stride: tl.constexpr,
    ROW_HEADS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK)
    address = ((row // ROW_HEADS) * x_stride_row
               + (row % ROW_HEADS) * x_stride_head
               + dims * x_stride_dim)
    hidden = tl.load(x_ptr + address, mask=dims < WIDTH, other=0).to(tl.float32)
    variance = tl.sum(hidden * hidden, axis=0) / WIDTH
    inverse_rms = tl.rsqrt(variance + eps)
    # Eager HF/nano's correctness seam casts the normalized FP32 stream back
    # to BF16 before multiplying the BF16 norm weight. Do not fuse that cast
    # away or move the weight into the FP32 normalization expression.
    rounded = (hidden * inverse_rms).to(x_ptr.dtype.element_ty).to(tl.float32)
    weight = tl.load(weight_ptr + dims * weight_stride,
                     mask=dims < WIDTH, other=0).to(tl.float32)
    result = rounded * weight
    tl.store(out_ptr + row * WIDTH + dims, result, mask=dims < WIDTH)


def _validate(x, weight, eps, output_dtype, num_warps):
    """Metadata-only validation: no downloads, tensor scans or CUDA sync."""
    if x.ndim not in (1, 2, 3):
        raise ValueError("RMSNorm input must have one, two or three dimensions")
    width = int(x.shape[-1])
    if not 1 <= width <= 8192:
        raise ValueError("RMSNorm width must be between 1 and 8192")
    if weight.ndim != 1 or int(weight.numel()) != width:
        raise ValueError("RMSNorm weight must be a vector matching the last dimension")
    if x.dtype not in (torch.bfloat16, torch.float32) or weight.dtype != x.dtype:
        raise ValueError("RMSNorm input and weight must share BF16 or FP32 dtype")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm eps must be a positive finite scalar")
    if isinstance(num_warps, bool) or num_warps not in (4, 8):
        raise ValueError("RMSNorm num_warps must be 4 or 8")
    if output_dtype not in (None, x.dtype, torch.float32):
        raise ValueError("RMSNorm output dtype must match input or be FP32")
    if not x.is_cuda or x.device != weight.device:
        raise ValueError("RMSNorm requires input and weight on one CUDA device")


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, *,
             output_dtype: torch.dtype | None = None, num_warps: int = 4) -> torch.Tensor:
    """Fuse reference-order RMSNorm with native BF16/FP32 operands.

    Supports noncontiguous leading rows/heads, strided feature dimensions and
    strided weights without copying the input. ``output_dtype=torch.float32``
    exposes the result *before the final store rounding*, still retaining the
    reference intermediate input-dtype cast. It never upcasts the operands to
    change the experiment. Empty leading dimensions produce an empty output.
    """
    _validate(x, weight, eps, output_dtype, num_warps)
    dtype = x.dtype if output_dtype is None else output_dtype
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    if x.numel() == 0:
        return out
    width = int(x.shape[-1])
    if x.ndim == 3:
        row_stride, head_stride, row_heads = x.stride(0), x.stride(1), int(x.shape[1])
    elif x.ndim == 2:
        row_stride, head_stride, row_heads = x.stride(0), 0, 1
    else:
        row_stride, head_stride, row_heads = 0, 0, 1
    _reference_order_rms_norm[(x.numel() // width,)](
        out, x, weight, float(eps),
        x_stride_row=row_stride, x_stride_head=head_stride,
        x_stride_dim=x.stride(-1), weight_stride=weight.stride(0),
        ROW_HEADS=row_heads, WIDTH=width, BLOCK=triton.next_power_of_2(width),
        num_warps=num_warps, enable_fp_fusion=False,
    )
    return out


def reference_rms_norm(x: torch.Tensor, norm) -> torch.Tensor:
    """Drop-in *explicit experimental* entry with the Qwen reference signature."""
    return rms_norm(x, norm.weight, norm.eps)
