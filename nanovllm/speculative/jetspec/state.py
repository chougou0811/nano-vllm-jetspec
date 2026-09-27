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
