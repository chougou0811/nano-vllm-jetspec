"""Exact-row CUDA graphs for serial packed Target verification.

Inspired by official JetSpec's GraphedVerify, but not its single-request tree
padding or compiled BF16 arithmetic. We capture the existing forward unchanged:
GEMMs still see sum(tree_nodes) real rows, attention reads live ragged metadata,
and only provisional scratch slots are written. Canonical commit stays outside.

Outputs are borrowed until the next verify. The synchronous runner consumes and
copies accepted features before that boundary. All graphs share one memory pool
and must never execute concurrently or retain outputs across rounds.
"""
from __future__ import annotations

from dataclasses import replace

import torch


_DEVICE_FIELDS = ("cu_seqlens_q", "query_to_request", "query_local_row",
                  "prefix_lens", "block_tables", "tree_slots", "qq_bias",
                  "qq_bias_offsets", "node_counts")


def graph_signature(metadata, *, prefix_backend: bool) -> tuple:
    """Only execution geometry matters; no request ID/prefix/slot is cached."""
    return (metadata.total_queries, len(metadata.prefix_lengths),
            metadata.qq_bias.numel(), bool(prefix_backend))


class PackedTargetGraph:
    """Runner-owned, bounded, exact-shape graph cache; no allocator page leases."""

    def __init__(self, target, kv_pool, target_layer_ids, max_model_len,
                 *, max_graphs=16):
        if not kv_pool.is_cuda:
            raise ValueError("Target CUDA Graph requires CUDA; no CPU fallback")
        if max_graphs < 1 or max_model_len < 1:
            raise ValueError("positive graph capacity and model length required")
        self.target, self.kv_pool = target, kv_pool
        self.target_layer_ids = tuple(target_layer_ids)
        self.table_width = (max_model_len + kv_pool.shape[3] - 1) // kv_pool.shape[3]
        self.max_graphs = int(max_graphs)
        self.entries = {}
        self.pool = None
        self.capture_stream = torch.cuda.Stream(device=kv_pool.device)
        self.ready = None
        self.captures = self.replays = self.eager_fallbacks = 0
        self.staged_bytes = 0
        self.pool_identity = self._pool_identity()

    def _pool_identity(self):
        return (self.kv_pool.data_ptr(), tuple(self.kv_pool.shape),
                tuple(self.kv_pool.stride()), self.kv_pool.dtype, self.kv_pool.device)

    def _forward(self, entry):
        hidden, taps = self.target.model.forward_packed_tree(
            entry["tokens"], entry["positions"], self.kv_pool,
            entry["metadata"], self.target_layer_ids)
        # Keep final hidden available for same-state numerical qualification.
        return self.target.lm_head(hidden), taps, hidden

    def _stage(self, entry, tokens, positions, metadata):
        entry["tokens"].copy_(tokens)
        entry["positions"].copy_(positions)
        staged = entry["metadata"]
        for name in _DEVICE_FIELDS:
            dst, src = getattr(staged, name), getattr(metadata, name)
            if name == "block_tables":
                # A fixed-width table avoids a new graph every 256 prefix tokens.
                # The kernel's actual prefix length never accesses padded entries.
                dst[:, :src.shape[1]].copy_(src)
            else:
                dst.copy_(src)
            self.staged_bytes += src.numel() * src.element_size()
        self.staged_bytes += tokens.numel() * (tokens.element_size() + positions.element_size())

    def _new_entry(self, tokens, positions, metadata):
        tensors = {name: torch.empty_like(getattr(metadata, name)) for name in _DEVICE_FIELDS
                   if name != "block_tables"}
        tensors["block_tables"] = torch.full(
            (len(metadata.prefix_lengths), self.table_width), -1,
            device=self.kv_pool.device, dtype=metadata.block_tables.dtype)
        staged = replace(metadata, **tensors)
        entry = {"tokens": torch.empty_like(tokens),
                 "positions": torch.empty_like(positions), "metadata": staged}
        self._stage(entry, tokens, positions, metadata)
        return entry

    def _record_ready(self):
        if self.ready is None:
            self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(self.kv_pool.device))

    @torch.inference_mode()
    def verify(self, tokens, positions, metadata):
        # Captured Python validation sees capture-time host metadata. Validate
        # every REAL round here before any device scatter, including graph hits.
        metadata.validate_pool_geometry(self.kv_pool)
        if tokens.ndim != 1 or tokens.shape != positions.shape or tokens.numel() != metadata.total_queries:
            raise ValueError("invalid packed graph token/position geometry")
        if tokens.device != self.kv_pool.device or positions.device != self.kv_pool.device:
            raise ValueError("packed graph inputs must share the live KV device")
        if self._pool_identity() != self.pool_identity:
            raise RuntimeError("Target graph's live KV pool changed")
        if metadata.block_tables.shape[1] > self.table_width:
            raise ValueError("packed prefix exceeds graph table capacity")
        if self.ready is not None:
            torch.cuda.current_stream(self.kv_pool.device).wait_event(self.ready)
        # Same qualified auto dispatch as eager. Prefix lengths remain device
        # values in the captured loops, NOT frozen capture-time scalar bounds.
        from nanovllm.speculative.jetspec.paged_backend import _use_prefix_tree_attention
        attn = self.target.model.layers[0].self_attn
        query_probe = self.kv_pool.new_empty((0, attn.num_heads, attn.head_dim))
        prefix = _use_prefix_tree_attention(query_probe, self.kv_pool[0, 0],
            self.kv_pool[1, 0], metadata, attn.num_heads // attn.num_kv_heads)
        key = graph_signature(metadata, prefix_backend=prefix)
        entry = self.entries.get(key)
        if entry is None and len(self.entries) >= self.max_graphs:
            # Finite memory: uncommon shapes execute the original eager path.
            # Never evict a pool while an earlier graph can reference it.
            self.eager_fallbacks += 1
            hidden, taps = self.target.model.forward_packed_tree(
                tokens, positions, self.kv_pool, metadata, self.target_layer_ids)
            result = self.target.lm_head(hidden), taps
            self.last_hidden = hidden
            self._record_ready()
            return result
        if entry is None:
            entry = self._new_entry(tokens, positions, metadata)
            # These writes target THIS transaction's scratch, never a zero/default
            # slot or a committed prefix. Warm-up compiles all lazy Triton kernels
            # before capture and exercises precisely the actual input shape.
            # Sharing a graph memory pool also requires one capture stream.
            stream = self.capture_stream
            stream.wait_stream(torch.cuda.current_stream(self.kv_pool.device))
            try:
                with torch.cuda.stream(stream):
                    self._forward(entry)
                    self._forward(entry)
                torch.cuda.current_stream(self.kv_pool.device).wait_stream(stream)
                torch.cuda.synchronize(self.kv_pool.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                    entry["outputs"] = self._forward(entry)
            except BaseException:
                # A CPU exception can leave queued scatter writes on the warmup
                # stream. Fence them BEFORE the transaction can release scratch.
                stream.synchronize()
                raise
            finally:
                torch.cuda.current_stream(self.kv_pool.device).wait_stream(stream)
            self.pool = graph.pool()
            entry["graph"] = graph
            self.entries[key] = entry
            self.captures += 1
        else:
            self._stage(entry, tokens, positions, metadata)
        entry["graph"].replay()
        self.replays += 1
        self._record_ready()
        self.last_hidden = entry["outputs"][2]
        return entry["outputs"][:2]

    def snapshot(self):
        return {"execution": "cuda_graph", "cached_graphs": len(self.entries),
                "max_graphs": self.max_graphs, "captures": self.captures,
                "replays": self.replays, "eager_fallbacks": self.eager_fallbacks,
                "staged_bytes": self.staged_bytes,
                "row_padding": False, "table_width": self.table_width}

    def close(self):
        if self.ready is not None:
            self.ready.synchronize()
        self.entries.clear()
        self.pool = self.ready = None
