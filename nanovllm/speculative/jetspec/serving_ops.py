"""Small host decisions with one batch transfer at each required boundary.

The tree policy is the official accum_logp heap. Target argmax stays on GPU;
the tiny request-local acceptance walk consumes one combined argmax download.
No full logits or historical features are downloaded by these operations.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch


def build_trees(roots, proposals, budgets, depth, width, device):
    """Run top-k in one batch and retain authoritative CPU tree topology."""
    from jetspec.tree.baselines.accum_logp import _build_from_topk

    active = [i for i, proposal in enumerate(proposals) if proposal is not None]
    topk = {}
    if active:
        logits = torch.cat([proposals[i] for i in active], dim=0)
        if logits.ndim != 3 or logits.shape[1] != depth:
            raise ValueError("one [1, depth, vocab] Draft proposal is required per active request")
        probabilities = torch.log_softmax(logits, dim=-1)
        values, tokens = torch.topk(probabilities, width, dim=-1)
        # Identical representation to the official builder, but one round-wide
        # host transfer instead of one transfer/synchronization per request.
        pairs = torch.stack((tokens.to(torch.float64), values.to(torch.float64)), -1).cpu().tolist()
        topk = dict(zip(active, pairs))
    cpu_trees = []
    for i, (root, budget) in enumerate(zip(roots, budgets)):
        pairs = topk.get(i, [])
        tree = _build_from_topk(int(root),
            [[int(pair[0]) for pair in row] for row in pairs],
            [[pair[1] for pair in row] for row in pairs], int(budget), torch.device("cpu"))
        cpu_trees.append(tree)
    # One H2D allocation for all integer topology; views keep request bounds.
    packed = torch.stack([torch.cat([getattr(tree, key) for tree in cpu_trees])
                          for key in ("token_ids", "parent_indices", "depth")]).to(device)
    trees, offset = [], 0
    for cpu in cpu_trees:
        end = offset + cpu.num_nodes
        trees.append(SimpleNamespace(token_ids=packed[0, offset:end],
            parent_indices=packed[1, offset:end], depth=packed[2, offset:end],
            num_nodes=cpu.num_nodes, child_maps=cpu.child_maps,
            host_token_ids=cpu.token_ids.tolist(), host_depths=cpu.depth.tolist(),
            host_parents=cpu.parent_indices.tolist(), host_ancestor=cpu.ancestor,
            # This is host metadata. The fast metadata constructor consumes it
            # directly; diagnostic callers can explicitly move it to device.
            ancestor=cpu.ancestor))
        offset = end
    return trees


def accept_batch(logits, trees, offsets, max_depth):
    """Greedy argmax once, one compact download, then independent CPU walks.

    child_maps preserve the official last-child-wins rule for duplicate sibling
    tokens. The resulting root-inclusive paths are host-authoritative inputs to
    the transaction; their GPU indices are constructed only at physical commit.
    """
    predicted = logits.argmax(-1).tolist()
    results = []
    for i, tree in enumerate(trees):
        lo, hi = offsets[i:i + 2]
        greedy = predicted[lo:hi]
        tokens = getattr(tree, "host_token_ids", None)
        depths = getattr(tree, "host_depths", None)
        children = getattr(tree, "child_maps", None)
        if tokens is None or depths is None or children is None:
            # Compatibility with externally supplied debug/test tree builders.
            # The production accum_logp path always carries host topology.
            from jetspec.tree._core.accept import _build_child_maps_cpu
            tokens = tree.token_ids.tolist()
            depths = tree.depth.tolist()
            parents = tree.parent_indices.tolist()
            children = _build_child_maps_cpu(tokens, parents, tree.num_nodes)
        path = [0]
        while len(path) <= max_depth:
            child = children[path[-1]].get(greedy[path[-1]])
            if child is None:
                break
            path.append(child)
        if [depths[node] for node in path] != list(range(len(path))):
            raise RuntimeError("accepted RoPE positions do not match canonical tail")
        correction = greedy[path[-1]]
        results.append({"path": path, "accepted_length": len(path) - 1,
                        "correction": correction,
                        "outputs": [tokens[node] for node in path[1:]] + [correction]})
    return results
