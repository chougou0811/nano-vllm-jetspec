"""Independent CPU FP64 diagnostics for the Phase-3.1 numerical contract.

These helpers never read production qq_bias, page tables, or tree-attention
code. The caller supplies chronological visible K/V assembled independently
from the canonical prefix and parent chain. Valid operands must be finite:
masked NaN/Inf activations are not covered by a finite-input isolation claim.

The fixed 1e-4 absolute-scaled and relative-RMS bounds apply to a pre-BF16
FP32 attention result, not to a BF16 result after quantization. BF16 ULP and
nearest-rounding diagnostics are reported separately rather than weakening
those bounds after seeing a model mismatch.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import torch


FP32_ATTENTION_BOUND = 1e-4


def parent_chain(parent_indices: Sequence[int] | torch.Tensor, chosen_node: int) -> list[int]:
    """Return root-to-node indices without consulting a precomputed mask.

    JetSpec's root is index 0 with parent -1. Reject a cyclic, disconnected,
    or out-of-bounds selected chain rather than silently omitting a bad key.
    Other branches are not traversed and do not affect this reconstruction.
    """
    if isinstance(parent_indices, torch.Tensor):
        if parent_indices.ndim != 1 or parent_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("parents must be a one-dimensional integer vector")
        parents = parent_indices.detach().cpu().tolist()
    else:
        parents = list(parent_indices)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in parents):
            raise ValueError("parents must contain integer indices")
    if not parents or not isinstance(chosen_node, int) or not 0 <= chosen_node < len(parents):
        raise ValueError("chosen node is outside the tree")
    if parents[0] != -1:
        raise ValueError("root parent must be -1")
    reverse = []
    visited = set()
    current = chosen_node
    while True:
        if current < 0 or current >= len(parents):
            raise ValueError("parent index is outside the tree")
        if current in visited:
            raise ValueError("cycle in selected parent chain")
        visited.add(current)
        reverse.append(current)
        parent = parents[current]
        if parent == -1:
            if current != 0:
                raise ValueError("selected chain is disconnected from root 0")
            return reverse[::-1]
        current = parent


def _cpu_fp64(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise ValueError(f"{name} must be an explicitly supplied CPU tensor")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    value = tensor.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains a nonfinite valid operand")
    return value


def fp64_attention(
    q: torch.Tensor,
    visible_k: torch.Tensor,
    visible_v: torch.Tensor,
    scale: float,
    num_queries_per_kv: int,
) -> torch.Tensor:
    """Unmasked one-query attention over ONLY independent visible keys.

    Shapes are q[Hq,D], K/V[L,Hkv,D]; output is CPU FP64[Hq,D]. GQA
    grouping is checked explicitly. The supplied key order should be the
    chronological prefix followed by root-to-chosen-node tree ancestors.
    """
    query = _cpu_fp64(q, "q")
    keys = _cpu_fp64(visible_k, "visible_k")
    values = _cpu_fp64(visible_v, "visible_v")
    if query.ndim != 2 or keys.ndim != 3 or keys.shape != values.shape:
        raise ValueError("expected q[Hq,D] and equal K/V[L,Hkv,D]")
    if keys.shape[0] == 0 or keys.shape[1] == 0 or keys.shape[2] != query.shape[1]:
        raise ValueError("visible key sequence and head dimensions must be nonempty and agree")
    if (not isinstance(num_queries_per_kv, int) or num_queries_per_kv < 1
            or query.shape[0] != keys.shape[1] * num_queries_per_kv):
        raise ValueError("invalid GQA head grouping")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("attention scale must be finite and positive")
    outputs = []
    # Deliberately direct per-head operations: no paged kernel, masks,
    # packed addressing, online-softmax tiles, or copied Triton arithmetic.
    for query_head in range(query.shape[0]):
        kv_head = query_head // num_queries_per_kv
        scores = torch.mv(keys[:, kv_head], query[query_head]) * float(scale)
        probabilities = torch.softmax(scores, dim=0)
        outputs.append(torch.matmul(probabilities, values[:, kv_head]))
    return torch.stack(outputs)


def numerical_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    bound_scale: float = FP32_ATTENTION_BOUND,
    bf16_output: bool | None = None,
) -> dict:
    """Preserve FP64 precision and report errors plus separate BF16 ULPs.

    ``within_fixed_bound`` is the conjunction of maxabs <= bound_scale *
    max(1, reference_absmax) and relative RMS <= bound_scale. The bound is
    applicable to FP32 pre-round attention output only; a BF16 output should
    be assessed with the separately reported quantization diagnostics.
    """
    left = _cpu_fp64(actual, "actual")
    right = _cpu_fp64(reference, "reference")
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError("metric tensors must have equal nonempty shapes")
    if not math.isfinite(float(bound_scale)) or float(bound_scale) <= 0:
        raise ValueError("bound scale must be finite and positive")
    quantized_output = actual.dtype == torch.bfloat16 if bf16_output is None else bool(bf16_output)
    difference = left - right
    max_abs = float(difference.abs().max())
    rms_error = float(difference.square().mean().sqrt())
    reference_rms = float(right.square().mean().sqrt())
    reference_absmax = float(right.abs().max())
    relative_rms = rms_error / reference_rms if reference_rms else (0.0 if rms_error == 0.0 else float("inf"))
    max_abs_bound = float(bound_scale) * max(1.0, reference_absmax)
    # BF16's adjacent spacing is asymmetric at powers of two. Report the
    # larger neighboring spacing explicitly; it is not a claimed error bound.
    nearest_bf16 = right.to(dtype=torch.bfloat16)
    if not bool(torch.isfinite(nearest_bf16).all()):
        raise ValueError("reference exceeds finite BF16 range")
    rounded = nearest_bf16.to(dtype=torch.float64)
    up = torch.nextafter(nearest_bf16, torch.full_like(nearest_bf16, float("inf"))).double()
    down = torch.nextafter(nearest_bf16, torch.full_like(nearest_bf16, float("-inf"))).double()
    up_spacing = (up - rounded).abs()
    down_spacing = (rounded - down).abs()
    # At BF16's largest finite value, one neighbor is infinity. Use its
    # finite neighbor spacing rather than hiding all error behind infinity.
    up_spacing = torch.where(torch.isfinite(up_spacing), up_spacing, down_spacing)
    down_spacing = torch.where(torch.isfinite(down_spacing), down_spacing, up_spacing)
    ulp = torch.maximum(up_spacing, down_spacing)
    bf16_difference = left.to(dtype=torch.bfloat16).double() - rounded
    return {
        "elements": left.numel(), "max_abs_error": max_abs,
        "rms_error": rms_error, "relative_rms_error": relative_rms,
        "reference_absmax": reference_absmax, "reference_rms": reference_rms,
        "max_abs_bound": max_abs_bound, "relative_rms_bound": float(bound_scale),
        "within_fixed_bound": max_abs <= max_abs_bound and relative_rms <= float(bound_scale),
        "fixed_bound_applicable": not quantized_output,
        "fixed_bound_scope": "FP32 attention output before BF16 quantization",
        "bf16_quantization": {
            "actual_output_is_bf16": quantized_output,
            "ulp_definition": "max adjacent spacing around FP64-oracle rounded BF16 value",
            "max_error_in_bf16_ulps": float((difference.abs() / ulp).max()),
            "max_distance_from_nearest_bf16_in_ulps": float((bf16_difference.abs() / ulp).max()),
            "nearest_bf16_unequal_elements": int((bf16_difference != 0).sum()),
            "nearest_rounding_max_abs_error": float((rounded - right).abs().max()),
        },
    }


def argmax_flip_witness(logits_a: torch.Tensor, logits_b: torch.Tensor) -> dict:
    """Report whether flips fit the exact sup-error/margin inequality.

    B is the reference. For delta=||A-B||_infinity, a reference winner with
    margin >2*delta cannot lose in A. This is a mathematical consistency
    witness, NOT evidence that delta itself is acceptably small or numerical.
    Ties use torch.argmax's first-index rule, not torch.topk's tie ordering.
    """
    actual = _cpu_fp64(logits_a, "logits_a")
    reference = _cpu_fp64(logits_b, "logits_b")
    if actual.shape != reference.shape or actual.ndim < 1 or actual.shape[-1] < 2:
        raise ValueError("logits must have equal shapes with vocabulary size >=2")
    actual = actual.reshape(-1, actual.shape[-1])
    reference = reference.reshape(-1, reference.shape[-1])
    if actual.shape[0] == 0:
        raise ValueError("logits must contain at least one row")
    rows = []
    flips = 0
    violations = 0
    for index, (left, right) in enumerate(zip(actual, reference)):
        winner_a = int(left.argmax())
        winner_b = int(right.argmax())
        reference_top = torch.topk(right, k=2).values
        margin = float(reference_top[0] - reference_top[1])
        delta = float((left - right).abs().max())
        flipped = winner_a != winner_b
        candidate_gap = float(right[winner_b] - right[winner_a])
        # The winner-vs-flipped-candidate gap is stronger than top1/top2 when
        # the losing reference candidate is below rank two.
        consistent = margin <= 2 * delta and candidate_gap <= 2 * delta
        violation = flipped and not consistent
        flips += int(flipped)
        violations += int(violation)
        rows.append({
            "row": index, "actual_argmax": winner_a, "reference_argmax": winner_b,
            "argmax_flipped": flipped, "logits_inf_delta": delta,
            "reference_top2_margin": margin, "two_delta": 2 * delta,
            "reference_winner_vs_actual_winner_gap": candidate_gap,
            "flip_within_error_envelope": (consistent if flipped else None),
            "large_margin_violation": violation,
            "actual_candidate_logits": [float(left[winner_a]), float(left[winner_b])],
            "reference_candidate_logits": [float(right[winner_a]), float(right[winner_b])],
        })
    return {"rows_checked": len(rows), "argmax_flips": flips,
            "large_margin_violations": violations, "all_flip_witnesses_consistent": violations == 0,
            "scope": "necessary inequality only; not a numerical correctness proof",
            "rows": rows}
