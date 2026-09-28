from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DenseTargetState:
    """Committed-only state for the Phase-1 correctness runtime."""

    committed: torch.Tensor
    key_values: list[tuple[torch.Tensor, torch.Tensor]]
    target_hidden: torch.Tensor

    @property
    def cache_len(self) -> int:
        return int(self.key_values[0][0].shape[0]) if self.key_values else 0

    def assert_round_invariant(self) -> None:
        expected = int(self.committed.shape[1]) - 1
        if self.cache_len != expected or int(self.target_hidden.shape[1]) != expected:
            raise RuntimeError(
                "JetSpec state invariant failed: "
                f"committed-1={expected}, cache={self.cache_len}, "
                f"target_hidden={self.target_hidden.shape[1]}"
            )

    def commit_tree_path(
        self,
        provisional_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        node_hidden: torch.Tensor,
        accepted_path: torch.Tensor,
    ) -> None:
        """Commit root + accepted nodes; rejected rows become unreachable immediately."""
        selected = accepted_path.to(dtype=torch.long)
        next_cache = []
        for (old_k, old_v), (node_k, node_v) in zip(self.key_values, provisional_key_values):
            next_cache.append((
                torch.cat((old_k, node_k.index_select(0, selected)), dim=0),
                torch.cat((old_v, node_v.index_select(0, selected)), dim=0),
            ))
        self.key_values = next_cache
        self.target_hidden = torch.cat(
            (self.target_hidden, node_hidden.index_select(1, selected)), dim=1
        )

    def clear(self) -> None:
        self.key_values.clear()
        self.target_hidden = torch.empty(0)
        self.committed = torch.empty(0, dtype=torch.long)


@dataclass
class PagedTargetState:
    """Logical committed sequence backed by nano-vLLM's physical KV pool."""

    committed: torch.Tensor
    target_hidden: torch.Tensor
    kv_pool: torch.Tensor
    block_manager: object
    block_size: int
    logical_slots: torch.Tensor
    owned_blocks: list[int]
    pending_blocks: list[int]

    @classmethod
    def from_prefill(
        cls,
        committed: torch.Tensor,
        prompt_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        target_hidden: torch.Tensor,
        kv_pool: torch.Tensor,
        block_manager,
        block_size: int,
    ) -> "PagedTargetState":
        cache_len = int(prompt_key_values[0][0].shape[0])
        num_blocks = (cache_len + block_size - 1) // block_size
        blocks = block_manager.reserve_provisional(num_blocks)
        block_tensor = torch.tensor(blocks, dtype=torch.long, device=kv_pool.device)
        positions = torch.arange(cache_len, dtype=torch.long, device=kv_pool.device)
        slots = block_tensor[positions // block_size] * block_size + positions % block_size
        physical_blocks = torch.div(slots, block_size, rounding_mode="floor").long()
        offsets = torch.remainder(slots, block_size).long()
        for layer_id, (keys, values) in enumerate(prompt_key_values):
            kv_pool[0, layer_id, physical_blocks, offsets] = keys
            kv_pool[1, layer_id, physical_blocks, offsets] = values
        return cls(
            committed=committed,
            target_hidden=target_hidden,
            kv_pool=kv_pool,
            block_manager=block_manager,
            block_size=int(block_size),
            logical_slots=slots,
            owned_blocks=blocks,
            pending_blocks=[],
        )

    @property
    def cache_len(self) -> int:
        return int(self.logical_slots.numel())

    def reserve_tree(self, n_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pending_blocks:
            raise RuntimeError("previous provisional tree allocation is still active")
        num_blocks = (int(n_nodes) + self.block_size - 1) // self.block_size
        self.pending_blocks = self.block_manager.reserve_provisional(num_blocks)
        block_tensor = torch.tensor(
            self.pending_blocks, dtype=torch.long, device=self.kv_pool.device
        )
        offsets = torch.arange(n_nodes, dtype=torch.long, device=self.kv_pool.device)
        node_slots = (
            block_tensor[offsets // self.block_size] * self.block_size
            + offsets % self.block_size
        )
        return node_slots, torch.cat((self.logical_slots, node_slots))

    def commit_tree_path(
        self,
        node_slots: torch.Tensor,
        node_hidden: torch.Tensor,
        accepted_path: torch.Tensor,
    ) -> dict[str, int]:
        if not self.pending_blocks:
            raise RuntimeError("no provisional tree allocation to commit")
        selected = accepted_path.long()
        accepted_slots = node_slots.index_select(0, selected)
        self.logical_slots = torch.cat((self.logical_slots, accepted_slots))
        self.target_hidden = torch.cat(
            (self.target_hidden, node_hidden.index_select(1, selected)), dim=1
        )
        # A nano block contains 256 slots while this tree has at most 63 nodes.
        # Root is always committed, so accepted and rejected nodes share a live
        # physical block. Drop rejected logical ownership now and retain the block
        # lease until request cleanup; token-granular free would corrupt accepted KV.
        reserved = len(self.pending_blocks)
        self.owned_blocks.extend(self.pending_blocks)
        self.pending_blocks = []
        return {
            "reserved_blocks": reserved,
            "released_blocks": 0,
            "rejected_logical_slots": int(node_slots.numel() - accepted_slots.numel()),
        }

    def assert_round_invariant(self) -> None:
        expected = int(self.committed.shape[1]) - 1
        if self.cache_len != expected or int(self.target_hidden.shape[1]) != expected:
            raise RuntimeError(
                "JetSpec paged state invariant failed: "
                f"committed-1={expected}, logical_slots={self.cache_len}, "
                f"target_hidden={self.target_hidden.shape[1]}"
            )
        if self.pending_blocks:
            raise RuntimeError("provisional blocks survived a completed round")
        for block_id in self.owned_blocks:
            block = self.block_manager.blocks[block_id]
            if block.ref_count != 1 or block_id not in self.block_manager.used_block_ids:
                raise RuntimeError(f"lost paged KV ownership for block {block_id}")

    def clear(self) -> int:
        blocks = self.pending_blocks + self.owned_blocks
        if blocks:
            self.block_manager.release_provisional(blocks)
        released = len(blocks)
        self.pending_blocks = []
        self.owned_blocks = []
        self.logical_slots = torch.empty(0, dtype=torch.long, device=self.kv_pool.device)
        self.target_hidden = torch.empty(0, device=self.kv_pool.device)
        self.committed = torch.empty(0, dtype=torch.long, device=self.kv_pool.device)
        return released
