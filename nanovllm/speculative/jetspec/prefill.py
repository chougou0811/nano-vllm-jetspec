"""Request-owned incremental prefill storage; no speculative scratch ownership.

Only ``processed_tokens`` rows are visible. Chunks append KV directly into private
canonical pages and taps into one capacity buffer. Promotion transfers ownership
to PagedTargetState rather than copying historical KV or concatenating taps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


def offset_causal_mask(prefix_length: int, chunk_length: int, device) -> torch.Tensor | None:
    """SDPA bool visibility for Q=C, K=P+C (True means visible).

    ``is_causal=True`` alone aligns a non-square mask to its upper-left corner,
    which is wrong for a chunk whose queries start at logical position P.
    """
    if prefix_length < 0 or chunk_length < 1:
        raise ValueError("invalid incremental attention lengths")
    if not prefix_length:
        return None  # Preserve the existing full-prefill execution shape/order.
    keys = torch.arange(prefix_length + chunk_length, device=device)
    queries = prefix_length + torch.arange(chunk_length, device=device)
    return keys.unsqueeze(0) <= queries.unsqueeze(1)


def canonical_slots(blocks: list[int], length: int, block_size: int, device) -> torch.Tensor:
    if length < 0 or len(blocks) * block_size < length:
        raise ValueError("canonical pages do not cover the requested length")
    if not length:
        return torch.empty(0, dtype=torch.long, device=device)
    pages = torch.tensor(blocks, dtype=torch.long, device=device)
    offsets = torch.arange(block_size, device=device)
    return (pages[:, None] * block_size + offsets[None, :]).flatten()[:length]


@dataclass(eq=False)
class PrefillContext:
    request_id: str | int
    tokens: torch.Tensor
    kv_pool: torch.Tensor
    block_manager: object
    block_size: int
    tree_budget: int
    max_new_tokens: int
    ignore_eos: bool
    prompt_length: int
    created_at: float
    snapshot: dict | None = None
    processed_tokens: int = 0
    owned_blocks: list[int] = field(default_factory=list)
    chunk_records: list[dict] = field(default_factory=list)
    cancelled: bool = False
    promoted: bool = False
    error: str | None = None
    feature_storage: torch.Tensor | None = field(default=None, repr=False)
    _ready: object | None = field(default=None, repr=False)
    _writing_stream: object | None = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def remaining_tokens(self) -> int:
        return self.total_tokens - self.processed_tokens

    @property
    def logical_slots(self) -> torch.Tensor:
        return canonical_slots(self.owned_blocks, self.processed_tokens,
                               self.block_size, self.kv_pool.device)

    @property
    def target_hidden(self) -> torch.Tensor | None:
        return None if self.feature_storage is None else self.feature_storage[:, :self.processed_tokens]

    def chunk_length(self, num_tokens: int) -> int:
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens < 1:
            raise ValueError("prefill chunk length must be a positive integer")
        if self.cancelled or self.promoted or self.remaining_tokens < 1:
            raise RuntimeError("prefill context is no longer writable")
        return min(num_tokens, self.remaining_tokens)

    def capacity(self, num_tokens: int) -> dict:
        count = self.chunk_length(num_tokens)
        pages = (self.processed_tokens + count + self.block_size - 1) // self.block_size
        needed = pages - len(self.owned_blocks)
        return {"num_tokens": count, "canonical_blocks": pages,
                "required_free_blocks": needed,
                "feasible": needed <= len(self.block_manager.free_block_ids)}

    def wait_ready(self) -> None:
        if self.kv_pool.is_cuda and self._ready is not None:
            torch.cuda.current_stream(self.kv_pool.device).wait_event(self._ready)

    def begin_writes(self) -> None:
        self.wait_ready()
        if self.kv_pool.is_cuda:
            self._writing_stream = torch.cuda.current_stream(self.kv_pool.device)

    def record_ready(self) -> None:
        if self.kv_pool.is_cuda:
            stream = self._writing_stream or torch.cuda.current_stream(self.kv_pool.device)
            event = torch.cuda.Event()
            event.record(stream)
            self._ready = event
        self._writing_stream = None

    def synchronize(self) -> None:
        if self.kv_pool.is_cuda:
            if self._ready is not None:
                self._ready.synchronize()
            # A failed forward/event recording may have queued writes which are
            # not covered by the last successful chunk's completion event.
            if self._writing_stream is not None:
                self._writing_stream.synchronize()

    def reserve(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        plan = self.capacity(num_tokens)
        new_blocks = self.block_manager.reserve_provisional(plan["required_free_blocks"])
        self.owned_blocks.extend(new_blocks)
        slots = canonical_slots(self.owned_blocks, self.processed_tokens + plan["num_tokens"],
                                self.block_size, self.kv_pool.device)
        return slots[:self.processed_tokens], slots[self.processed_tokens:]

    def append_features(self, taps: torch.Tensor, num_tokens: int) -> None:
        if taps.ndim != 2 or taps.shape[0] != num_tokens or taps.device != self.kv_pool.device:
            raise ValueError("prefill taps must have shape [chunk_tokens, feature_width]")
        if self.feature_storage is None:
            self.feature_storage = torch.empty((1, self.total_tokens, taps.shape[1]),
                dtype=taps.dtype, device=taps.device)
        elif (taps.shape[1] != self.feature_storage.shape[2] or
              taps.dtype != self.feature_storage.dtype):
            raise ValueError("prefill feature geometry changed between chunks")
        self.feature_storage[:, self.processed_tokens:self.processed_tokens + num_tokens].copy_(taps)

    def clear(self) -> int:
        if self.promoted or self.cancelled:
            return 0
        self.synchronize()
        released = len(self.owned_blocks)
        self.block_manager.release_provisional(self.owned_blocks)
        self.owned_blocks = []
        self.feature_storage = None
        self._ready = self._writing_stream = None
        self.cancelled = True
        return released


def forward_chunk(target, context: PrefillContext, num_tokens: int, target_layer_ids,
                  *, attention_backend="sdpa"):
    """Production paged incremental seam; compatibility fallback for CPU mocks."""
    prefix_slots, new_slots = context.reserve(num_tokens)
    start, count = context.processed_tokens, int(new_slots.numel())
    ids = context.tokens[0, start:start + count]
    positions = torch.arange(start, start + count, device=ids.device)
    context.begin_writes()
    incremental = getattr(target.model, "forward_dense_chunk", None)
    if incremental is not None:
        hidden, taps = incremental(ids, positions, context.kv_pool, prefix_slots,
                                   new_slots, target_layer_ids,
                                   **({"attention_backend": attention_backend}
                                      if attention_backend != "sdpa" else {}))
        backend = ("paged_layerwise_flash_chunk" if attention_backend == "flash_attn"
                   else "paged_layerwise_dense_chunk")
    else:
        if attention_backend != "sdpa":
            raise ValueError("Flash chunked prefill requires the real Qwen3 incremental seam")
        # Existing lightweight fake Targets expose only forward_dense. Real
        # Qwen3 never takes this all-layer dense-history compatibility path.
        pages, offsets = prefix_slots // context.block_size, prefix_slots % context.block_size
        past = ([(context.kv_pool[0, layer, pages, offsets],
                  context.kv_pool[1, layer, pages, offsets])
                 for layer in range(context.kv_pool.shape[1])] if start else None)
        hidden, new_kv, taps = target.model.forward_dense(ids, positions, past,
            offset_causal_mask(start, count, ids.device), target_layer_ids)
        if len(new_kv) != context.kv_pool.shape[1]:
            raise ValueError("incremental prefill KV layer count differs from pool")
        pages, offsets = new_slots // context.block_size, new_slots % context.block_size
        expected = (count, *context.kv_pool.shape[4:])
        for layer, (keys, values) in enumerate(new_kv):
            if tuple(keys.shape) != expected or tuple(values.shape) != expected:
                raise ValueError("incremental prefill KV shape differs from chunk/pool")
            context.kv_pool[0, layer, pages, offsets] = keys
            context.kv_pool[1, layer, pages, offsets] = values
        backend = "dense_mock_compatibility"
    context.append_features(taps, count)
    context.record_ready()
    context.processed_tokens += count
    kv_bytes = context.kv_pool.element_size() * context.kv_pool.shape[4] * context.kv_pool.shape[5]
    context.chunk_records.append({"prefix_tokens": start, "num_tokens": count,
        "processed_tokens": context.processed_tokens, "backend": backend,
        "prefix_gather_bytes_per_layer": 2 * start * kv_bytes,
        "feature_append_bytes": taps.numel() * taps.element_size()})
    return hidden
