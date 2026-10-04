"""Cross-request DFlash proposal batching with request-owned, compact KV.

The official head is reused unchanged. A group has one real head forward, with
rectangular old-prefix/new-suffix buffers and an explicit per-row visibility
mask. Only real context keys are published back into each request's official
DynamicCache; padding and speculative noise keys never survive the proposal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import DynamicCache
from jetspec.draft_head_adapter import DraftHeadTreeDrafter


@dataclass
class _Row:
    index: int
    request: Any
    forward: Any
    context_length: int
    cached_length: int
    past: tuple

    @property
    def suffix_length(self) -> int:
        return self.context_length - self.cached_length


class _RaggedCache:
    """Ephemeral Cache protocol used only during one official head forward.

    Old request caches are read-only. update() returns the rectangular batch
    consumed by SDPA and stages independently allocated compact row caches.
    Publishing is deferred until head, lm_head and every cache construction
    succeed, so a failed batched proposal cannot partly mutate persistent KV.
    """

    def __init__(self, rows: list[_Row], suffix_capacity: int):
        self.rows = rows
        self.prefix_capacity = max(row.cached_length for row in rows)
        self.suffix_capacity = suffix_capacity
        self.pending: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # The caller disables the head's automatic rectangular causal mask.
        # Real per-row lengths and block causality are in our explicit mask.
        return self.prefix_capacity

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx in self.pending:
            raise RuntimeError("a Draft layer updated its grouped cache twice")
        batch, heads, _, dim = key_states.shape
        if batch != len(self.rows) or value_states.shape != key_states.shape:
            raise ValueError("unexpected batched Draft KV geometry")
        compact = []
        if self.prefix_capacity:
            shape = (batch, heads, self.prefix_capacity, dim)
            prefix_k = key_states.new_zeros(shape)
            prefix_v = value_states.new_zeros(shape)
        for i, row in enumerate(self.rows):
            suffix_k = key_states[i:i + 1, :, :row.suffix_length]
            suffix_v = value_states[i:i + 1, :, :row.suffix_length]
            if row.cached_length:
                old_k, old_v = row.past[layer_idx]
                prefix_k[i:i + 1, :, :row.cached_length].copy_(old_k)
                prefix_v[i:i + 1, :, :row.cached_length].copy_(old_v)
                new_k = torch.cat((old_k, suffix_k), dim=-2)
                new_v = torch.cat((old_v, suffix_v), dim=-2)
            else:
                # A row view would retain every other request's batch storage
                # after cancellation. Explicit clones keep ownership compact.
                new_k = suffix_k.clone().contiguous()
                new_v = suffix_v.clone().contiguous()
            compact.append((new_k, new_v))
        self.pending[layer_idx] = compact
        if self.prefix_capacity:
            return (torch.cat((prefix_k, key_states), dim=-2),
                    torch.cat((prefix_v, value_states), dim=-2))
        return key_states, value_states

    def compact_caches(self, num_layers: int) -> list[DynamicCache]:
        if set(self.pending) != set(range(num_layers)):
            raise RuntimeError("not every Draft layer staged its compact cache")
        return [DynamicCache.from_legacy_cache(tuple(
            self.pending[layer][row] for layer in range(num_layers)))
            for row in range(len(self.rows))]


class BatchedDraftProposer:
    """Genuine grouped Draft forwards, with honest serial ablation/fallback.

    propose() accepts runtime requests (state.committed, state.target_hidden,
    drafter), and returns one (1, depth, vocab) tensor per input request in input
    order. Singleton groups and unsupported drafter/cache/backend types use the
    original per-request proposer. No model weights or canonical Target state
    are changed. This class is synchronous, like the surrounding eager runner.

    Groups cap both new-suffix projection and attention-key padding inflation.
    For example, a new long prompt is not padded across many warm short suffixes
    just to force one launch. This is bounded ragged padding, not a paged Draft
    attention kernel or a shared mutable Draft cache.
    """

    def __init__(self, head, target, *, enabled: bool = True, max_padding_ratio: float = 2.0):
        if (isinstance(max_padding_ratio, bool) or not isinstance(max_padding_ratio, (int, float)) or
                not 1 <= max_padding_ratio < float("inf")):
            raise ValueError("max_padding_ratio must be finite and at least one")
        self.head = head
        self.target = target
        self.enabled = bool(enabled)
        self.max_padding_ratio = float(max_padding_ratio)
        self.block_size = int(head.block_size)
        if self.block_size < 2:
            raise ValueError("Draft block_size must include at least one prediction")
        self.last_stats: dict = {}

    def _row(self, index, request) -> _Row | None:
        if type(request.drafter) is not DraftHeadTreeDrafter:
            return None
        forward = getattr(request.drafter, "_fwd", None)
        cache = getattr(forward, "cache", None)
        rope_type = getattr(self.head.rotary_emb, "rope_type", None)
        # Only the audited official eager SDPA/eager head/cache combination is
        # adapted. In particular Mock/proxy/custom drafters are not fake-batched.
        if (not isinstance(cache, DynamicCache) or
                getattr(forward, "head", None) is not self.head or
                getattr(forward, "target", None) is not self.target or
                getattr(forward, "block_size", None) != self.block_size or
                not isinstance(getattr(forward, "draft_shift", None), bool) or
                getattr(self.head.config, "_attn_implementation", None) not in ("eager", "sdpa") or
                self.head.training or not isinstance(rope_type, str) or
                "dynamic" in rope_type or rope_type == "longrope" or
                bool(getattr(cache, "offloading", False)) or
                any(layer.self_attn.sliding_window is not None for layer in self.head.layers)):
            return None
        hidden = request.state.target_hidden
        committed = request.state.committed
        if (hidden.ndim != 3 or hidden.shape[0] != 1 or committed.ndim != 2 or
                committed.shape[0] != 1 or committed.shape[1] != hidden.shape[1] + 1):
            raise ValueError("Draft inputs require one uncached anchor and aligned target taps")
        context_length = int(hidden.shape[1])
        cached_length = cache.get_seq_length()
        if not isinstance(cached_length, int):
            return None
        # Match the official proposer reset when a restored/shorter state is
        # presented; do not mutate its cache until the new proposal succeeds.
        if cached_length > context_length:
            cached_length, past = 0, ()
        else:
            past = cache.to_legacy_cache() if cached_length else ()
        if cached_length:
            if len(past) != len(self.head.layers):
                return None
            for keys, values in past:
                if (keys is None or values is None or keys.ndim != 4 or
                        keys.shape != values.shape or keys.shape[0] != 1 or
                        keys.shape[-2] != cached_length or keys.device != forward.device or
                        keys.dtype != forward.dtype):
                    return None
        return _Row(index, request, forward, context_length, cached_length, past)

    def _padding_ratios(self, rows):
        batch = len(rows)
        suffix = batch * max(1, max(row.suffix_length for row in rows))
        suffix /= sum(max(1, row.suffix_length) for row in rows)
        rectangular_keys = batch * (max(row.cached_length for row in rows) +
                                   max(row.suffix_length for row in rows) + self.block_size)
        real_keys = sum(row.context_length + self.block_size for row in rows)
        return suffix, rectangular_keys / real_keys

    def _groups(self, rows):
        # Similar suffix sizes prevent fresh/recomputed prefixes from inflating
        # all warm rows. Preserve result alignment through each row's index.
        ordered = sorted(rows, key=lambda row: (row.suffix_length, row.context_length))
        groups, group = [], []
        for row in ordered:
            candidate = group + [row]
            shifts = {item.forward.draft_shift for item in candidate}
            if group and (len(shifts) != 1 or max(self._padding_ratios(candidate)) > self.max_padding_ratio):
                groups.append(group)
                group = [row]
            else:
                group = candidate
        if group:
            groups.append(group)
        return groups

    def _batch(self, rows, depth):
        device, dtype = rows[0].forward.device, rows[0].forward.dtype
        suffix_capacity = max(row.suffix_length for row in rows)
        cache = _RaggedCache(rows, suffix_capacity)
        target_dim = int(rows[0].request.state.target_hidden.shape[-1])
        taps = torch.zeros((len(rows), suffix_capacity, target_dim), device=device, dtype=dtype)
        for i, row in enumerate(rows):
            suffix = row.request.state.target_hidden[:, row.cached_length:]
            if suffix.shape[-1] != target_dim:
                raise ValueError("batched target taps have inconsistent feature dimensions")
            taps[i:i + 1, :row.suffix_length].copy_(suffix)
        anchors = torch.cat([row.request.state.committed[:, -1:] for row in rows]).to(device)
        placeholders = torch.full((len(rows), self.block_size - 1), self.head.mask_token_id,
                                  dtype=anchors.dtype, device=device)
        noise = self.target.model.embed_tokens(torch.cat((anchors, placeholders), dim=1))
        cached = torch.tensor([row.cached_length for row in rows], device=device)
        lengths = torch.tensor([row.context_length for row in rows], device=device)
        deltas = lengths - cached
        suffix_positions = cached[:, None] + torch.arange(suffix_capacity, device=device)[None, :]
        real_suffix = torch.arange(suffix_capacity, device=device)[None, :] < deltas[:, None]
        suffix_positions = torch.where(real_suffix, suffix_positions, 0)
        block_positions = lengths[:, None] + torch.arange(self.block_size, device=device)[None, :]
        positions = torch.cat((suffix_positions, block_positions), dim=1)
        prefix_valid = torch.arange(cache.prefix_capacity, device=device)[None, :] < cached[:, None]
        context_valid = torch.cat((prefix_valid, real_suffix), dim=1)
        query = torch.arange(self.block_size, device=device)[:, None]
        key = torch.arange(self.block_size, device=device)[None, :]
        block_valid = key <= query if self.head.resolve_causal_head("auto") else torch.ones(
            (self.block_size, self.block_size), device=device, dtype=torch.bool)
        allowed = torch.cat((context_valid[:, None, :].expand(-1, self.block_size, -1),
                             block_valid[None, :, :].expand(len(rows), -1, -1)), dim=-1)
        mask = torch.zeros(allowed.shape, device=device, dtype=dtype).masked_fill_(
            ~allowed, torch.finfo(dtype).min).unsqueeze(1)
        hidden = self.head(target_hidden=taps, noise_embedding=noise, position_ids=positions,
                           attention_mask=mask, past_key_values=cache, use_cache=True,
                           is_causal=False)
        shift = rows[0].forward.draft_shift
        draft_slice = slice(0, self.block_size - 1) if shift else slice(1, self.block_size)
        logits = self.target.lm_head(hidden[:, draft_slice])[:, :depth]
        caches = cache.compact_caches(len(self.head.layers))
        pad_slots = len(rows) * (cache.prefix_capacity + suffix_capacity) - sum(
            row.context_length for row in rows)
        kv_padding_bytes = sum(pad_slots * pair[0].shape[1] * pair[0].shape[-1] *
                               pair[0].element_size() * 2 for pair in caches[0])
        return logits, caches, kv_padding_bytes

    @staticmethod
    def _cache_bytes(requests):
        total = 0
        for request in requests:
            cache = getattr(getattr(request.drafter, "_fwd", None), "cache", None)
            if isinstance(cache, DynamicCache):
                for keys, values in cache:
                    if keys is not None:
                        total += keys.numel() * keys.element_size()
                    if values is not None:
                        total += values.numel() * values.element_size()
        return total

    @staticmethod
    def _cache_storage_bytes(requests):
        # Serial DynamicCache.crop() keeps a view that can retain a few noise
        # slots. Distinguish its actual backing storage from logical KV bytes.
        storages = {}
        for request in requests:
            cache = getattr(getattr(request.drafter, "_fwd", None), "cache", None)
            if isinstance(cache, DynamicCache):
                for pair in cache:
                    for tensor in pair:
                        if tensor is not None:
                            storage = tensor.untyped_storage()
                            storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
        return sum(storages.values())

    @torch.inference_mode()
    def propose(self, requests, depth: int | None = None) -> list[torch.Tensor]:
        requests = list(requests)
        depth = self.block_size - 1 if depth is None else depth
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth < self.block_size:
            raise ValueError("depth must fit the trained Draft block")
        if len({id(request) for request in requests}) != len(requests):
            raise ValueError("duplicate request in Draft proposal batch")
        stats = {"enabled": self.enabled, "batched_forward_calls": 0, "serial_forward_calls": 0,
                 "batch_sizes": [], "max_padding_ratio": 1.0, "transient_padding_slots": 0,
                 "transient_kv_padding_bytes": 0,
                 "cache_bytes_before": self._cache_bytes(requests), "cache_bytes_after": 0,
                 "cache_storage_bytes_before": self._cache_storage_bytes(requests),
                 "cache_storage_bytes_after": 0}
        self.last_stats = stats
        output = [None] * len(requests)
        rows, serial = [], []
        for index, request in enumerate(requests):
            row = self._row(index, request) if self.enabled else None
            (rows if row is not None else serial).append(row if row is not None else (index, request))
        pending = []
        for group in self._groups(rows):
            if len(group) == 1:
                serial.append((group[0].index, group[0].request))
                continue
            stats["batched_forward_calls"] += 1
            stats["batch_sizes"].append(len(group))
            stats["max_padding_ratio"] = max(stats["max_padding_ratio"], *self._padding_ratios(group))
            stats["transient_padding_slots"] += len(group) * (
                max(row.cached_length for row in group) + max(row.suffix_length for row in group)
            ) - sum(row.context_length for row in group)
            logits, caches, padding_bytes = self._batch(group, depth)
            stats["transient_kv_padding_bytes"] += padding_bytes
            for i, (row, cache) in enumerate(zip(group, caches)):
                output[row.index] = logits[i:i + 1]
                pending.append((row.forward, cache))
        for index, request in sorted(serial, key=lambda item: item[0]):
            stats["serial_forward_calls"] += 1
            output[index] = request.drafter.propose_logits(
                request.state.committed, depth, target_hidden=request.state.target_hidden)
        # Publishing grouped caches cannot expose noise/pad keys or row views.
        for forward, cache in pending:
            forward.cache = cache
        stats["cache_bytes_after"] = self._cache_bytes(requests)
        stats["cache_storage_bytes_after"] = self._cache_storage_bytes(requests)
        return output

    proposal = propose
