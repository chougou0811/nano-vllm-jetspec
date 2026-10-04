"""Diagnostic-only instrumentation of the actual Qwen3 Target execution.

This module does not implement a second forward. It temporarily wraps the real
forward methods/operators and copies chosen logical rows to CPU. Use only in an
otherwise idle TP1 process: ``torch.nn.functional`` patches are process-global.
All methods/operators are restored even if Target execution raises.

Example::

    with LayerTrace(target, selected_rows=[node_row], capture_full_layers=(0,)) as t:
        hidden, taps = target.model.forward_packed_tree(ids, positions, pool, meta)
        logits = target.compute_logits(hidden)
    comparison = compare_traces(packed_trace, compact_path_trace)

``full_records`` is opt-in and contains CPU copies of all query rows in selected
layers. It permits causal replay of one GEMM or attention operator without
retaining 36 layers' full activations on the GPU. Production code is untouched.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import AbstractContextManager
from types import MethodType
from typing import Iterable

import torch
import torch.nn.functional as F

from nanovllm.models import qwen3
from nanovllm.speculative.jetspec import paged_backend


def tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict:
    """Element-bitwise and FP32 error metrics, explicitly retaining nonfinite info."""
    a, b = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape),
                "dtype_a": str(a.dtype), "dtype_b": str(b.dtype),
                "bitwise_equal": False, "incompatible": True}
    byte_differences = a.reshape(-1).view(torch.uint8).reshape(-1, a.element_size()) != b.reshape(
        -1).view(torch.uint8).reshape(-1, b.element_size())
    unequal = int(byte_differences.any(dim=1).sum().item())
    af, bf = a.float(), b.float()
    finite = torch.isfinite(af) & torch.isfinite(bf)
    delta = (af - bf)[finite]
    rms = float(delta.square().mean().sqrt().item()) if delta.numel() else None
    scale = float(bf[finite].square().mean().sqrt().item()) if delta.numel() else None
    return {"shape": list(a.shape), "dtype": str(a.dtype), "elements": a.numel(),
            "bitwise_equal": unequal == 0, "unequal_elements": unequal,
            "unequal_fraction": unequal / max(1, a.numel()),
            "max_abs": float(delta.abs().max().item()) if delta.numel() else None,
            "rms": rms, "relative_rms": rms / max(scale, 1e-30) if rms is not None else None,
            "nonfinite_a": int((~torch.isfinite(af)).sum().item()),
            "nonfinite_b": int((~torch.isfinite(bf)).sum().item())}


class LayerTrace(AbstractContextManager):
    """Capture selected token rows from actual packed, paged, or dense forwards.

    Rows must refer to token dimension zero after any reference-path remapping.
    ``records`` is ordered in execution order, so its first unequal entry is the
    earliest observed divergence rather than a hand-picked later logit delta.
    ``stage_inputs`` stores linear inputs for selected rows, useful for verifying
    that a projection first diverged with exactly equal operands.
    """

    def __init__(self, target, selected_rows: Iterable[int], *,
                 capture_full_layers: Iterable[int] = ()):
        self.target = target
        self.model = target.model
        self.selected_rows = tuple(int(row) for row in selected_rows)
        if not self.selected_rows or min(self.selected_rows) < 0:
            raise ValueError("at least one nonnegative selected token row is required")
        self.capture_full_layers = frozenset(int(i) for i in capture_full_layers)
        self.records = OrderedDict()
        self.full_records = OrderedDict()
        self.stage_inputs = OrderedDict()
        self.operator_shapes = OrderedDict()
        self._restorations = []
        self._layer = None
        self._entered = False
        self._silu_depth = 0
        self._linear_stages = {}
        for layer_id, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            qw = attn.qkv_proj.weight
            widths = (attn.q_size, attn.kv_size, attn.kv_size)
            offset = 0
            for name, width in zip(("q_raw", "k_raw", "v_raw"), widths):
                self._linear_stages[self._weight_key(qw[offset:offset + width])] = (layer_id, name)
                offset += width
            self._linear_stages[self._weight_key(attn.o_proj.weight)] = (layer_id, "o_proj")
            width = layer.mlp.gate_up_proj.output_sizes[0]
            mw = layer.mlp.gate_up_proj.weight
            self._linear_stages[self._weight_key(mw[:width])] = (layer_id, "mlp_gate")
            self._linear_stages[self._weight_key(mw[width:])] = (layer_id, "mlp_up")
            self._linear_stages[self._weight_key(layer.mlp.down_proj.weight)] = (layer_id, "mlp_down")
        self._linear_stages[self._weight_key(target.lm_head.weight)] = (None, "lm_head")

    @staticmethod
    def _weight_key(weight):
        return (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()))

    @staticmethod
    def _key(layer_id, stage):
        return stage if layer_id is None else f"layer_{layer_id:02d}.{stage}"

    def _rows(self, value):
        if value.ndim < 1 or max(self.selected_rows) >= value.shape[0]:
            raise ValueError(f"selected trace rows {self.selected_rows} exceed tensor {tuple(value.shape)}")
        rows = torch.tensor(self.selected_rows, device=value.device, dtype=torch.long)
        return value.index_select(0, rows).detach().to("cpu").clone()

    def _capture(self, stage, value, *, layer_id=None):
        key = self._key(layer_id, stage)
        if key in self.records:
            raise RuntimeError(f"trace stage executed twice: {key}; use a fresh trace for each forward")
        self.records[key] = self._rows(value)
        if layer_id in self.capture_full_layers:
            self.full_records[key] = value.detach().to("cpu").clone()

    def _replace(self, owner, name, replacement):
        # Instances inherit bound forwards; delete our replacement to restore the
        # original class descriptor instead of permanently shadowing that method.
        is_module = isinstance(owner, torch.nn.Module)
        existed = name in owner.__dict__ if is_module else True
        original = getattr(owner, name)
        self._restorations.append((owner, name, original, existed))
        setattr(owner, name, replacement)

    def __enter__(self):
        if self._entered:
            raise RuntimeError("a LayerTrace cannot be entered twice")
        self._entered = True
        try:
            original_linear = F.linear

            def linear(x, weight, bias=None):
                stage = self._linear_stages.get(self._weight_key(weight))
                if stage is None:
                    return original_linear(x, weight, bias)
                layer_id, name = stage
                key = self._key(layer_id, name)
                self.stage_inputs[key] = self._rows(x)
                self.operator_shapes[key] = {"input": list(x.shape), "weight": list(weight.shape),
                                             "bias": None if bias is None else list(bias.shape),
                                             "input_stride": list(x.stride()),
                                             "weight_stride": list(weight.stride())}
                if name == "o_proj":
                    self._capture("o_proj_input", x, layer_id=layer_id)
                elif name == "mlp_down":
                    self._capture("mlp_activation", x, layer_id=layer_id)
                elif name == "lm_head":
                    self._capture("lm_head_input", x)
                if layer_id in self.capture_full_layers:
                    self.full_records[key + "_input"] = x.detach().to("cpu").clone()
                result = original_linear(x, weight, bias)
                self._capture(name, result, layer_id=layer_id)
                return result

            self._replace(F, "linear", linear)
            original_norm = qwen3._reference_rms_norm
            norm_stages = {id(self.model.norm): (None, "final_hidden")}
            for i, layer in enumerate(self.model.layers):
                norm_stages[id(layer.input_layernorm)] = (i, "input_norm")
                norm_stages[id(layer.post_attention_layernorm)] = (i, "post_attention_norm")
                if not layer.self_attn.qkv_bias:
                    norm_stages[id(layer.self_attn.q_norm)] = (i, "q_norm")
                    norm_stages[id(layer.self_attn.k_norm)] = (i, "k_norm")

            def rms_norm(x, norm):
                stage = norm_stages.get(id(norm))
                if stage is not None:
                    layer_id, name = stage
                    if name == "post_attention_norm":
                        self._capture("post_attention_residual", x, layer_id=layer_id)
                    elif name == "final_hidden":
                        self._capture("final_norm_input", x)
                result = original_norm(x, norm)
                if stage is not None:
                    self._capture(name, result, layer_id=layer_id)
                return result

            self._replace(qwen3, "_reference_rms_norm", rms_norm)
            original_silu = F.silu

            def silu(x, *args, **kwargs):
                # ModelRunner enables a default-device TorchFunctionMode. The
                # original Python F.silu dispatches through its module-global
                # ``silu`` symbol, which now points at this wrapper, so it can
                # re-enter us once with that mode disabled. Observe the outer
                # call only; do not suppress/replace PyTorch's actual dispatch.
                outermost = self._silu_depth == 0
                self._silu_depth += 1
                try:
                    result = original_silu(x, *args, **kwargs)
                finally:
                    self._silu_depth -= 1
                if outermost and self._layer is not None:
                    self._capture("mlp_silu", result, layer_id=self._layer)
                return result

            self._replace(F, "silu", silu)
            original_packed = paged_backend.packed_tree_attention

            def packed_attention(q, k_pool, v_pool, metadata, *args, **kwargs):
                layer_id = self._layer
                slots = metadata.tree_slots
                blocks = torch.div(slots, metadata.block_size, rounding_mode="floor").long()
                offsets = torch.remainder(slots, metadata.block_size).long()
                self._capture("q_rope", q, layer_id=layer_id)
                self._capture("k_rope", k_pool[blocks, offsets], layer_id=layer_id)
                self._capture("v_scattered", v_pool[blocks, offsets], layer_id=layer_id)
                result = original_packed(q, k_pool, v_pool, metadata, *args, **kwargs)
                self._capture("attention_output", result, layer_id=layer_id)
                return result

            self._replace(paged_backend, "packed_tree_attention", packed_attention)
            original_paged = paged_backend.paged_tree_attention

            def paged_attention(q, k_pages, v_pages, *args, **kwargs):
                # The legacy seam passes slots as logical_slot_map at arg 8.
                # Rotated Q and attention output alone suffice for c1 comparison;
                # full K/V is captured by packed traces for causal operator replay.
                self._capture("q_rope", q, layer_id=self._layer)
                result = original_paged(q, k_pages, v_pages, *args, **kwargs)
                self._capture("attention_output", result, layer_id=self._layer)
                return result

            self._replace(paged_backend, "paged_tree_attention", paged_attention)
            original_sdpa = F.scaled_dot_product_attention

            def sdpa(q, k, v, *args, **kwargs):
                if self._layer is not None:
                    self._capture("q_rope", q.squeeze(0).transpose(0, 1), layer_id=self._layer)
                result = original_sdpa(q, k, v, *args, **kwargs)
                if self._layer is not None:
                    self._capture("attention_output", result.squeeze(0).transpose(0, 1),
                                  layer_id=self._layer)
                return result

            self._replace(F, "scaled_dot_product_attention", sdpa)
            for layer_id, layer in enumerate(self.model.layers):
                for method in ("forward_packed_tree", "forward_paged_tree", "forward_dense"):
                    original = getattr(layer, method)

                    def layer_forward(_self, *args, _original=original, _layer_id=layer_id, **kwargs):
                        # Every seam receives (positions, hidden_states, ...).
                        previous = self._layer
                        self._layer = _layer_id
                        try:
                            hidden = args[1] if len(args) > 1 else kwargs["hidden_states"]
                            self._capture("input_hidden", hidden, layer_id=_layer_id)
                            result = _original(*args, **kwargs)
                            output = result[0] if isinstance(result, tuple) else result
                            self._capture("output_hidden", output, layer_id=_layer_id)
                            return result
                        finally:
                            self._layer = previous

                    self._replace(layer, method, MethodType(layer_forward, layer))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, name, original, existed in reversed(self._restorations):
            if existed:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self._restorations.clear()
        self._layer = None
        self._silu_depth = 0
        return False


def compare_traces(left: LayerTrace, right: LayerTrace, *, include_equal: bool = True) -> dict:
    """Compare matching logical rows, rejecting incomplete/reordered stage schemas.

    ``earliest_divergence`` remains the first observed *numeric* difference in
    matching stages. Missing, additional or reordered observations are reported
    independently; they must never silently satisfy a fixed-layout equality gate.
    Operator input/shape names are part of the schema, but execution shape values
    deliberately need not match when diagnosing different packed shapes.
    """
    left_order, right_order = list(left.records), list(right.records)
    schema_equal = (left_order == right_order
                    and list(left.stage_inputs) == list(right.stage_inputs)
                    and list(left.operator_shapes) == list(right.operator_shapes))
    stages = OrderedDict()
    earliest = None
    for name, value in left.records.items():
        if name not in right.records:
            stages[name] = {"missing_in_reference": True}
            continue
        metrics = tensor_metrics(value, right.records[name])
        if name in left.stage_inputs and name in right.stage_inputs:
            metrics["operator_input"] = tensor_metrics(left.stage_inputs[name], right.stage_inputs[name])
            metrics["operator_shape_a"] = left.operator_shapes.get(name)
            metrics["operator_shape_b"] = right.operator_shapes.get(name)
        if not metrics["bitwise_equal"] and earliest is None:
            earliest = {"stage": name, **metrics}
        if include_equal or not metrics["bitwise_equal"]:
            stages[name] = metrics
    missing_left = [name for name in right.records if name not in left.records]
    missing_right = [name for name in left.records if name not in right.records]
    return {"selected_rows_a": list(left.selected_rows), "selected_rows_b": list(right.selected_rows),
            "compared_stages": sum(name in right.records for name in left.records),
            "earliest_divergence": earliest, "stages": stages,
            "trace_schema_equal": schema_equal,
            "stage_order_a": left_order, "stage_order_b": right_order,
            "missing_reference_stages": missing_right,
            "reference_only_stages": missing_left,
            "all_compared_stages_bitwise_equal": schema_equal and earliest is None}


def logit_margin(logits: torch.Tensor, *, topk: int = 5) -> dict:
    """CPU-friendly top-logit report; BF16 ties remain visible after FP32 cast."""
    row = logits.detach().float().cpu().reshape(-1)
    values, ids = torch.topk(row, min(topk, row.numel()))
    maximum = row.max()
    tied = torch.nonzero(row == maximum, as_tuple=False).flatten()
    return {"argmax": int(row.argmax().item()),
            "top_ids": ids.tolist(), "top_values": values.tolist(),
            "top1_top2_margin": float((values[0] - values[1]).item()) if values.numel() > 1 else None,
            "max_tie_ids": tied.tolist()}


def replay_packed_attention_fp32(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    metadata,
    scale: float,
    num_queries_per_kv: int,
) -> torch.Tensor:
    """Run the *same production kernel* but retain FP32 output before rounding.

    Q/K/V dtypes, operands, physical layout, masks, launch grid and TILE64 are
    unchanged. Only the destination pointer's dtype differs. As a compiler
    control, callers must also compare ``result.to(q.dtype)`` with an ordinary
    ``packed_tree_attention`` replay using these exact operands. If that fails,
    the FP32-output variant is not evidence about the production accumulator.
    This is diagnostic instrumentation, not a serving backend or alternate
    implementation of online softmax.
    """
    import triton

    if q.ndim != 3 or q.shape[0] != metadata.total_queries or not q.is_cuda:
        raise ValueError("FP32 replay requires CUDA queries matching packed metadata")
    if k_pool.ndim != 4 or k_pool.shape != v_pool.shape or k_pool.shape[1] != metadata.block_size:
        raise ValueError("FP32 replay KV geometry does not match metadata")
    if q.device != k_pool.device or q.device != v_pool.device or q.device != metadata.tree_slots.device:
        raise ValueError("FP32 replay operands must be on one device")
    total_q, heads, head_size = q.shape
    if q.stride(-1) != 1 or k_pool.shape[-1] != head_size or heads != k_pool.shape[2] * num_queries_per_kv:
        raise ValueError("unsupported FP32 replay head geometry")
    out = torch.empty_like(q, dtype=torch.float32)
    paged_backend._packed_paged_tree_fp32[(total_q, heads)](
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
