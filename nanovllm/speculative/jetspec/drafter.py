"""Official JetSpec draft-head adapter for nano-vLLM's target weights.

The imported components are engine-independent code from pinned JetSpec commit
2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f.  nano-vLLM owns target execution;
the official checkpoint model and its causal-parallel cache semantics stay intact.
"""

from __future__ import annotations

import torch


def load_official_drafter(draft_model: str, target, tree_depth: int):
    try:
        from jetspec import DraftHeadTreeDrafter, load_draft_head
    except ImportError as exc:
        raise RuntimeError(
            "Phase-1 JetSpec runtime requires the pinned official JetSpec package "
            "to load its draft-head implementation"
        ) from exc

    head = load_draft_head(
        draft_model, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if int(head.block_size) != int(tree_depth) + 1:
        raise ValueError(
            f"draft block_size={head.block_size} does not match tree_depth={tree_depth}"
        )
    drafter = DraftHeadTreeDrafter(
        head,
        target=target,
        block_size=head.block_size,
        target_layer_ids=head.target_layer_ids,
        draft_shift=False,
    )
    return head, drafter
