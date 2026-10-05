"""Opt-in FlashAttention Draft proposals without changing the official head.

Each request contributes compact ``[old context, new context, noise block]``
keys to one varlen attention call per layer. Queries contain only the noise
block. FlashAttention >= 2.1's bottom-right causal alignment is consequently
exactly the official DFlash block-causal mask; noncausal heads use causal=False.
No padding key, noise key or another request's storage survives publication.

This adapter reuses the official head's projection, norms, RoPE, output
projection and MLP modules/weights. Only attention dispatch and ephemeral KV
layout change. The default SDPA proposer and the vendored JetSpec repository
are untouched. Explicit FlashAttention selection fails on unsupported heads
or unavailable kernels instead of silently producing an SDPA benchmark.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from packaging.version import Version
from transformers import DynamicCache

from jetspec.models.draft_head import apply_rotary_pos_emb
from nanovllm.speculative.jetspec.batched_draft import BatchedDraftProposer


def _load_flash_varlen():
    try:
        import flash_attn
        from flash_attn import flash_attn_varlen_func
    except ImportError as exc:
        raise RuntimeError("Flash Draft requires the external flash-attn package") from exc
    if Version(flash_attn.__version__) < Version("2.1"):
        raise RuntimeError("Flash Draft requires flash-attn >= 2.1 for bottom-right causal masking")
    return flash_attn_varlen_func


class FlashDraftProposer(BatchedDraftProposer):
    """Grouped varlen Draft forwards, including genuine Flash singletons.

    ``flash_varlen`` is a test-only dependency injection seam: CPU tests can
    validate packed geometry and masking against an independent FP32 oracle.
    Production construction omits it and always loads the external CUDA API.
    Grouping continues to bound padded *new-context projection* work, while
    attention keys are unpadded regardless of request history lengths.
    """

    def __init__(self, head, target, *, max_padding_ratio: float = 2.0,
                 flash_varlen: Callable | None = None):
        super().__init__(head, target, enabled=True, max_padding_ratio=max_padding_ratio)
        self._test_kernel = flash_varlen is not None
        self.flash_varlen = _load_flash_varlen() if flash_varlen is None else flash_varlen

    def _flash_group(self, rows, depth):
        device, dtype = rows[0].forward.device, rows[0].forward.dtype
        if not self._test_kernel and (device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16)):
            raise RuntimeError("Flash Draft requires CUDA FP16/BF16 tensors")
        batch, block_size = len(rows), self.block_size
        suffix_capacity = max(row.suffix_length for row in rows)
        target_dim = int(rows[0].request.state.target_hidden.shape[-1])
        taps = torch.zeros((batch, suffix_capacity, target_dim), device=device, dtype=dtype)
        for i, row in enumerate(rows):
            suffix = row.request.state.target_hidden[:, row.cached_length:]
            if suffix.shape[-1] != target_dim:
                raise ValueError("batched target taps have inconsistent feature dimensions")
            taps[i:i + 1, :row.suffix_length].copy_(suffix)
        anchors = torch.cat([row.request.state.committed[:, -1:] for row in rows]).to(device)
        placeholders = torch.full((batch, block_size - 1), self.head.mask_token_id,
                                  dtype=anchors.dtype, device=device)
        hidden = self.target.model.embed_tokens(torch.cat((anchors, placeholders), dim=1))
        projected_taps = self.head.hidden_norm(self.head.fc(self.head.hidden_dim_adapter(taps)))
        cached = torch.tensor([row.cached_length for row in rows], device=device)
        lengths = torch.tensor([row.context_length for row in rows], device=device)
        suffix_indices = torch.arange(suffix_capacity, device=device)[None, :]
        real_suffix = suffix_indices < (lengths - cached)[:, None]
        suffix_positions = torch.where(real_suffix, cached[:, None] + suffix_indices, 0)
        block_positions = lengths[:, None] + torch.arange(block_size, device=device)[None, :]
        position_ids = torch.cat((suffix_positions, block_positions), dim=1)
        position_embeddings = self.head.rotary_emb(hidden, position_ids)
        cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device) * block_size
        key_lengths = [row.context_length + block_size for row in rows]
        key_offsets = [0]
        for length in key_lengths:
            key_offsets.append(key_offsets[-1] + length)
        cu_k = torch.tensor(key_offsets, dtype=torch.int32, device=device)
        causal = self.head.resolve_causal_head("auto")
        pending = [[] for _ in rows]
        padding_bytes = 0

        for layer_index, layer in enumerate(self.head.layers):
            attn = layer.self_attn
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            q = attn.q_proj(normalized).view(batch, block_size, -1, attn.head_dim)
            q = attn.q_norm(q).transpose(1, 2)
            k_ctx, k_noise = attn.k_proj(projected_taps), attn.k_proj(normalized)
            v_ctx, v_noise = attn.v_proj(projected_taps), attn.v_proj(normalized)
            k = torch.cat((k_ctx, k_noise), dim=1).view(batch, suffix_capacity + block_size, -1, attn.head_dim)
            v = torch.cat((v_ctx, v_noise), dim=1).view(batch, suffix_capacity + block_size, -1, attn.head_dim)
            k = attn.k_norm(k).transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
            packed_k, packed_v = [], []
            for i, row in enumerate(rows):
                new_k = k[i:i + 1, :, :row.suffix_length]
                new_v = v[i:i + 1, :, :row.suffix_length]
                if row.cached_length:
                    old_k, old_v = row.past[layer_index]
                    compact_k = torch.cat((old_k, new_k), dim=-2)
                    compact_v = torch.cat((old_v, new_v), dim=-2)
                else:
                    # Even an empty/new short row must not retain batch storage.
                    compact_k = new_k.clone().contiguous()
                    compact_v = new_v.clone().contiguous()
                pending[i].append((compact_k, compact_v))
                packed_k.extend((compact_k[0].transpose(0, 1), k[i, :, suffix_capacity:].transpose(0, 1)))
                packed_v.extend((compact_v[0].transpose(0, 1), v[i, :, suffix_capacity:].transpose(0, 1)))
            q_flat = q.transpose(1, 2).contiguous().flatten(0, 1)
            output = self.flash_varlen(
                q_flat, torch.cat(packed_k, dim=0), torch.cat(packed_v, dim=0),
                cu_q, cu_k, block_size, max(key_lengths), dropout_p=0.0,
                softmax_scale=attn.scaling, causal=causal,
            )
            output = output.reshape(batch, block_size, -1)
            hidden = residual + attn.o_proj(output)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            pad_slots = batch * suffix_capacity - sum(row.suffix_length for row in rows)
            padding_bytes += pad_slots * k.shape[1] * attn.head_dim * k.element_size() * 2

        hidden = self.head.norm(hidden)
        shift = rows[0].forward.draft_shift
        draft_slice = slice(0, block_size - 1) if shift else slice(1, block_size)
        logits = self.target.lm_head(hidden[:, draft_slice])[:, :depth]
        # Construction may allocate or fail: finish all caches before publishing.
        caches = [DynamicCache.from_legacy_cache(tuple(layers)) for layers in pending]
        return logits, caches, padding_bytes

    @torch.inference_mode()
    def propose(self, requests, depth: int | None = None):
        requests = list(requests)
        depth = self.block_size - 1 if depth is None else depth
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth < self.block_size:
            raise ValueError("depth must fit the trained Draft block")
        if len({id(request) for request in requests}) != len(requests):
            raise ValueError("duplicate request in Draft proposal batch")
        stats = {"enabled": True, "attention_backend": "flash_attn_varlen",
                 "batched_forward_calls": 0, "serial_forward_calls": 0,
                 "singleton_forward_calls": 0, "flash_head_forward_calls": 0,
                 "flash_attention_calls": 0, "batch_sizes": [],
                 "max_padding_ratio": 1.0, "transient_padding_slots": 0,
                 "transient_kv_padding_bytes": 0, "attention_key_padding_slots": 0,
                 "causal_head": bool(self.head.resolve_causal_head("auto")),
                 "cache_bytes_before": self._cache_bytes(requests), "cache_bytes_after": 0,
                 "cache_storage_bytes_before": self._cache_storage_bytes(requests),
                 "cache_storage_bytes_after": 0}
        self.last_stats = stats
        rows = []
        for index, request in enumerate(requests):
            row = self._row(index, request)
            if row is None:
                raise RuntimeError("Flash Draft requires the supported official eager/SDPA inference head and DynamicCache")
            rows.append(row)
        output, pending = [None] * len(requests), []
        for group in self._groups(rows):
            stats["flash_head_forward_calls"] += 1
            stats["flash_attention_calls"] += len(self.head.layers)
            stats["batched_forward_calls"] += int(len(group) > 1)
            stats["singleton_forward_calls"] += int(len(group) == 1)
            stats["batch_sizes"].append(len(group))
            stats["max_padding_ratio"] = max(stats["max_padding_ratio"], *self._padding_ratios(group))
            stats["transient_padding_slots"] += len(group) * max(row.suffix_length for row in group) - sum(
                row.suffix_length for row in group)
            logits, caches, padding_bytes = self._flash_group(group, depth)
            stats["transient_kv_padding_bytes"] += padding_bytes
            for i, (row, cache) in enumerate(zip(group, caches)):
                output[row.index] = logits[i:i + 1]
                pending.append((row.forward, cache))
        for forward, cache in pending:
            forward.cache = cache
        stats["cache_bytes_after"] = self._cache_bytes(requests)
        stats["cache_storage_bytes_after"] = self._cache_storage_bytes(requests)
        return output

    proposal = propose
