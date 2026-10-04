"""Validated request-local metadata for one genuine packed tree verification.

Only tree nodes are packed. Each request keeps its own canonical prefix pages,
RoPE positions, ancestor matrix and eventual acceptance decision. The metadata
contains O(sum(prefix pages) + sum(tree nodes) + sum(tree nodes squared)) entries,
not a full-history slot map or a total-query-square cross-request mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class PackedTreeMetadata:
    cu_seqlens_q: torch.Tensor
    query_to_request: torch.Tensor
    query_local_row: torch.Tensor
    prefix_lens: torch.Tensor
    block_tables: torch.Tensor
    tree_slots: torch.Tensor
    qq_bias: torch.Tensor
    qq_bias_offsets: torch.Tensor
    node_counts: torch.Tensor
    block_size: int
    prefix_lengths: tuple[int, ...]
    query_offsets: tuple[int, ...]
    node_counts_host: tuple[int, ...]
    block_tables_host: tuple[tuple[int, ...], ...]
    tree_slot_ids: tuple[int, ...]
    request_ids: tuple[str | int, ...]

    @classmethod
    def build(
        cls,
        prefix_lens: Sequence[int],
        block_tables: Sequence[Sequence[int]],
        tree_slots: Sequence[torch.Tensor],
        ancestor_masks: Sequence[torch.Tensor],
        block_size: int,
        *,
        request_ids: Sequence[str | int] | None = None,
    ) -> "PackedTreeMetadata":
        """Validate on the host once, then construct compact device buffers.

        Scratch-page ownership/admission belongs to BatchTreeTransaction. This
        boundary additionally rejects overlapping scratch ranges, malformed
        ancestor masks and any scratch page shared with a live prefix.
        """
        prefixes = tuple(int(length) for length in prefix_lens)
        count = len(prefixes)
        if count < 1 or not (len(block_tables) == len(tree_slots) == len(ancestor_masks) == count):
            raise ValueError("packed metadata requires equally sized nonempty request lists")
        block_size = int(block_size)
        if block_size <= 0 or any(length < 0 for length in prefixes):
            raise ValueError("invalid packed prefix length or page geometry")
        ids = tuple(range(count)) if request_ids is None else tuple(request_ids)
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError("request IDs must be unique and match the request count")
        device = tree_slots[0].device
        tables = []
        nodes = []
        offsets = [0]
        bias_offsets = [0]
        query_requests = []
        local_rows = []
        biases = []
        all_slots = []
        live_pages = set()
        for request_index, (prefix_len, table, slots, ancestors) in enumerate(
                zip(prefixes, block_tables, tree_slots, ancestor_masks)):
            pages = tuple(int(page) for page in table)
            required = (prefix_len + block_size - 1) // block_size
            if len(pages) != required or len(set(pages)) != len(pages) or any(page < 0 for page in pages):
                raise ValueError("prefix block table must be a tight canonical page table")
            if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64) or slots.device != device:
                raise ValueError("tree slots must be integer vectors on one common device")
            n = int(slots.numel())
            if n < 1 or ancestors.shape != (n, n) or ancestors.dtype != torch.bool or ancestors.device != device:
                raise ValueError("ancestor masks must be request-local boolean [nodes,nodes] matrices")
            if not bool(ancestors.diagonal().all().item()) or bool(torch.triu(ancestors, diagonal=1).any().item()):
                raise ValueError("ancestor mask must include self and be causal in local flat order")
            slot_ids = tuple(int(slot) for slot in slots.tolist())
            if any(slot < 0 for slot in slot_ids):
                raise ValueError("scratch slot IDs cannot be negative")
            all_slots.extend(slot_ids)
            live_pages.update(pages)
            tables.append(pages)
            nodes.append(n)
            offsets.append(offsets[-1] + n)
            bias_offsets.append(bias_offsets[-1] + n * n)
            query_requests.extend([request_index] * n)
            local_rows.extend(range(n))
            biases.append(torch.where(
                ancestors,
                torch.zeros((), dtype=torch.float32, device=device),
                torch.full((), float("-inf"), dtype=torch.float32, device=device),
            ).reshape(-1))
        if len(set(all_slots)) != len(all_slots):
            raise ValueError("packed requests have overlapping scratch slot ranges")
        if any(slot // block_size in live_pages for slot in all_slots):
            raise ValueError("scratch pages overlap a live committed prefix")
        # A zero-prefix request still gets one harmless, masked table column;
        # kernel loads are masked by key_pos < prefix_len, not by padding values.
        width = max(1, max(len(table) for table in tables))
        padded_tables = [list(table) + [-1] * (width - len(table)) for table in tables]
        return cls(
            cu_seqlens_q=torch.tensor(offsets, dtype=torch.int32, device=device),
            query_to_request=torch.tensor(query_requests, dtype=torch.int32, device=device),
            query_local_row=torch.tensor(local_rows, dtype=torch.int32, device=device),
            prefix_lens=torch.tensor(prefixes, dtype=torch.int32, device=device),
            block_tables=torch.tensor(padded_tables, dtype=torch.int32, device=device),
            tree_slots=torch.cat([slots.to(dtype=torch.int64) for slots in tree_slots]).contiguous(),
            qq_bias=torch.cat(biases).contiguous(),
            qq_bias_offsets=torch.tensor(bias_offsets, dtype=torch.int64, device=device),
            node_counts=torch.tensor(nodes, dtype=torch.int32, device=device),
            block_size=block_size,
            prefix_lengths=prefixes,
            query_offsets=tuple(offsets),
            node_counts_host=tuple(nodes),
            block_tables_host=tuple(tables),
            tree_slot_ids=tuple(all_slots),
            request_ids=ids,
        )

    @property
    def num_requests(self) -> int:
        return len(self.prefix_lengths)

    @property
    def total_queries(self) -> int:
        return self.query_offsets[-1]

    @property
    def mask_elements(self) -> int:
        return int(self.qq_bias.numel())

    def request_slice(self, index: int) -> slice:
        if index < 0 or index >= self.num_requests:
            raise IndexError("packed request index is out of range")
        return slice(self.query_offsets[index], self.query_offsets[index + 1])

    def validate_pool_geometry(self, kv_pool: torch.Tensor) -> None:
        if kv_pool.ndim != 6 or kv_pool.shape[0] != 2 or kv_pool.shape[3] != self.block_size:
            raise ValueError("packed metadata and KV pool page geometry differ")
        if kv_pool.device != self.tree_slots.device:
            raise ValueError("packed metadata and KV pool must be on the same device")
        num_blocks = int(kv_pool.shape[2])
        if any(page >= num_blocks for table in self.block_tables_host for page in table):
            raise ValueError("prefix page lies outside the KV pool")
        if any(slot >= num_blocks * self.block_size for slot in self.tree_slot_ids):
            raise ValueError("tree slot lies outside the KV pool")

    def report(self) -> dict:
        tensors = (self.cu_seqlens_q, self.query_to_request, self.query_local_row,
                   self.prefix_lens, self.block_tables, self.tree_slots, self.qq_bias,
                   self.qq_bias_offsets, self.node_counts)
        return {"requests": self.num_requests, "total_queries": self.total_queries,
                "cu_seqlens_q": list(self.query_offsets), "prefix_lengths": list(self.prefix_lengths),
                "node_counts": list(self.node_counts_host), "mask_elements": self.mask_elements,
                "total_query_square_not_allocated": self.total_queries ** 2,
                "device_metadata_bytes": sum(t.numel() * t.element_size() for t in tensors),
                "block_size": self.block_size, "numerical_contract": "fp32_online_softmax_tile64"}
