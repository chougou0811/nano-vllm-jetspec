from __future__ import annotations

import time
from typing import Any

import torch

from nanovllm.speculative.jetspec.drafter import load_official_drafter
from nanovllm.speculative.jetspec.state import DenseTargetState, PagedTargetState
from nanovllm.utils.context import reset_context


class JetSpecRuntime:
    """Single-request, greedy, eager JetSpec execution.

    Tree construction and acceptance reuse engine-independent code from pinned
    JetSpec commit 2c7b3fa. Target execution uses nano-vLLM Qwen3 weights and an
    dense SDPA or paged tree seam. The paged path commits accepted raw KV into
    canonical request pages and retires reusable scratch. No Scheduler, prefix
    cache, CUDA graph or TP path is involved.
    """

    def __init__(self, target, tokenizer, draft_model: str, *, tree_depth: int = 15,
                 tree_width: int = 7, tree_budget: int = 63, kv_pool=None,
                 block_manager=None, block_size: int = 256):
        if tree_depth != 15 or tree_width != 7 or tree_budget != 63:
            raise ValueError("Phase-1 MVP is fixed to depth=15, width=7, budget=63")
        self.target = target
        self.tokenizer = tokenizer
        self.tree_depth = int(tree_depth)
        self.tree_width = int(tree_width)
        self.tree_budget = int(tree_budget)
        self.kv_pool = kv_pool
        self.block_manager = block_manager
        self.block_size = int(block_size)
        self.head, self.drafter = load_official_drafter(draft_model, target, tree_depth)
        self.target_layer_ids = tuple(int(i) for i in self.head.target_layer_ids)
        if self.target_layer_ids != (1, 9, 17, 25, 33):
            raise ValueError(f"unexpected target taps: {self.target_layer_ids}")
        from jetspec.tree import get_algorithm

        self.tree_algorithm = get_algorithm("accum_logp")
        self.eos_token_ids = self._resolve_eos()
        self._active_state: DenseTargetState | PagedTargetState | None = None

    def _resolve_eos(self) -> set[int]:
        ids = set()
        for value in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(getattr(self.target, "generation_config", None), "eos_token_id", None),
        ):
            if value is None:
                continue
            ids.update(int(x) for x in value) if isinstance(value, (list, tuple, set)) else ids.add(int(value))
        return ids

    @staticmethod
    def _causal_mask(query_len: int, prefix_len: int, device) -> torch.Tensor:
        qi = torch.arange(query_len, device=device).unsqueeze(1)
        kj = torch.arange(prefix_len + query_len, device=device).unsqueeze(0)
        return kj <= (prefix_len + qi)

    def _target_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None,
        attention_mask: torch.Tensor | None,
        capture_hidden: bool,
    ):
        hidden, new_kv, tapped = self.target.model.forward_dense(
            input_ids,
            positions,
            past_key_values,
            attention_mask,
            self.target_layer_ids if capture_hidden else (),
        )
        logits = self.target.lm_head(hidden)
        return logits, new_kv, tapped

    def _target_forward_paged(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        node_slots: torch.Tensor,
        logical_slots: torch.Tensor,
        qq_bias: torch.Tensor,
    ):
        if isinstance(self._active_state, PagedTargetState) and self._active_state.scratch_active:
            self._active_state.scratch.check_stream()
        hidden, tapped = self.target.model.forward_paged_tree(
            input_ids,
            positions,
            self.kv_pool,
            node_slots,
            logical_slots,
            qq_bias,
            self.block_size,
            self.target_layer_ids,
        )
        return self.target.lm_head(hidden), tapped, hidden

    @torch.inference_mode()
    def generate_target_paged(self, prompt: str | list[int], max_new_tokens: int = 32) -> dict:
        if self._active_state is not None:
            raise RuntimeError("JetSpecRuntime already has an active request")
        try:
            return self._generate_target_paged_request(prompt, max_new_tokens)
        finally:
            try:
                if self._active_state is not None:
                    self._active_state.clear()
            finally:
                self._active_state = None
                reset_context()

    def _generate_target_paged_request(self, prompt, max_new_tokens) -> dict:
        """Greedy comparator with the tree verify's fixed 63-row numerical shape.

        Row zero is the real next token; all other rows are isolated dummy nodes.
        This preserves QKV GEMM and attention reduction shapes of the tree backend.
        """
        reset_context()
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        input_ids = self._prompt_ids(prompt)
        device = input_ids.device
        prompt_len = int(input_ids.shape[1])
        logits, prompt_kv, prompt_hidden = self._target_forward(
            input_ids[0], torch.arange(prompt_len, device=device), None, None, True
        )
        first = logits[-1].argmax().view(1)
        state = PagedTargetState.from_prefill(
            torch.cat((input_ids, first.view(1, 1)), dim=1),
            prompt_kv, prompt_hidden.unsqueeze(0), self.kv_pool,
            self.block_manager, self.block_size,
        )
        self._active_state = state
        output = [int(first.item())]
        tree_nodes = self.tree_budget
        dummy = torch.zeros((tree_nodes - 1,), dtype=torch.long, device=device)
        isolated = torch.eye(tree_nodes, dtype=torch.bool, device=device)
        qq_bias = torch.where(
            isolated,
            torch.zeros((), dtype=torch.float32, device=device),
            torch.full((), float("-inf"), dtype=torch.float32, device=device),
        )
        try:
            while len(output) < max_new_tokens and output[-1] not in self.eos_token_ids:
                state.assert_round_invariant()
                node_slots, logical_slots = state.reserve_tree(tree_nodes, max_path_length=1)
                positions = torch.cat((
                    torch.tensor([state.cache_len], device=device),
                    torch.full((tree_nodes - 1,), state.cache_len + 1, device=device),
                ))
                step_logits, node_hidden, _ = self._target_forward_paged(
                    torch.cat((state.committed[0, -1:].contiguous(), dummy)),
                    positions, node_slots, logical_slots, qq_bias,
                )
                next_token = step_logits[0].argmax().view(1)
                state.commit_tree_path(
                    node_slots, node_hidden.unsqueeze(0),
                    torch.zeros((1,), dtype=torch.long, device=device),
                    committed_tokens=torch.cat((state.committed, next_token.view(1, 1)), dim=1),
                )
                output.append(int(next_token.item()))
                state.assert_round_invariant()
        finally:
            released = state.clear()
            state.clear()  # cleanup must be idempotent
            self._active_state = None
        return {
            "token_ids": output,
            "numerical_path": "padded_single_path_63",
            "blocks_released_cleanup": released,
            "allocator_used_blocks_after": len(self.block_manager.used_block_ids),
        }

    @torch.inference_mode()
    def _qualification_probe(self, tree, state: PagedTargetState, ancestor,
                             positions, node_slots, paged_logits, paged_taps,
                             paged_final_hidden) -> dict:
        """Compare one frozen paged state/tree with dense and branchwise verifies."""
        from jetspec.tree import gpu_tree_accept

        n = int(tree.num_nodes)
        prefix_len = state.cache_len
        pool = state.kv_pool
        slots = state.logical_slots
        blocks = torch.div(slots, state.block_size, rounding_mode="floor").long()
        offsets = torch.remainder(slots, state.block_size).long()
        prefix_kv = [
            (pool[0, layer_id, blocks, offsets], pool[1, layer_id, blocks, offsets])
            for layer_id in range(len(self.target.model.layers))
        ]
        allowed = torch.ones((n, prefix_len + n), dtype=torch.bool, device=slots.device)
        allowed[:, prefix_len:] = ancestor
        dense_final, _, dense_taps = self.target.model.forward_dense(
            tree.token_ids, positions, prefix_kv, allowed, self.target_layer_ids
        )
        dense_logits = self.target.lm_head(dense_final)
        dense_greedy = dense_logits.argmax(dim=-1)
        paged_greedy = paged_logits.argmax(dim=-1)
        first_flip = torch.nonzero(dense_greedy != paged_greedy).flatten()
        flip_index = int(first_flip[0].item()) if first_flip.numel() else None
        dense_path, dense_accepted, dense_correction = gpu_tree_accept(
            tree.token_ids, dense_greedy, tree.parent_indices, tree.depth,
            max_depth=self.tree_depth,
        )
        paged_path, paged_accepted, paged_correction = gpu_tree_accept(
            tree.token_ids, paged_greedy, tree.parent_indices, tree.depth,
            max_depth=self.tree_depth,
        )

        def decision(path, accepted, correction):
            return {
                "path_indices": [int(x) for x in path.tolist()],
                "accepted_length": int(accepted),
                "correction_token_id": int(correction.item()),
                "path_token_ids": [int(x) for x in tree.token_ids.index_select(0, path).tolist()],
            }

        def tensor_diff(a, b):
            delta = (a.float() - b.float()).abs()
            return {"max_abs": float(delta.max().item()), "mean_abs": float(delta.mean().item())}

        singleton_blocks = self.block_manager.reserve_provisional(1)
        try:
            singleton_slot = torch.tensor(
                [singleton_blocks[0] * self.block_size],
                dtype=torch.long, device=slots.device,
            )
            singleton_logits, singleton_taps, singleton_final = self._target_forward_paged(
                tree.token_ids[:1], positions[:1], singleton_slot,
                torch.cat((slots, singleton_slot)),
                torch.zeros((1, 1), dtype=torch.float32, device=slots.device),
            )
            singleton_values, singleton_ids = singleton_logits[0].float().topk(2)
            singleton_root = {
                "argmax_token_id": int(singleton_logits[0].argmax().item()),
                "top2_token_ids": [int(x) for x in singleton_ids.tolist()],
                "top2_logits": [float(x) for x in singleton_values.tolist()],
                "paged_tree_root_argmax_token_id": int(paged_greedy[0].item()),
                "tap_diff": tensor_diff(singleton_taps[0], paged_taps[0]),
                "final_hidden_diff": tensor_diff(singleton_final[0], paged_final_hidden[0]),
                "logits_diff": tensor_diff(singleton_logits[0], paged_logits[0]),
            }
        finally:
            torch.cuda.current_stream(slots.device).synchronize()
            self.block_manager.release_provisional(singleton_blocks)

        margin = None
        if flip_index is not None:
            margin = {}
            for label, logits in (("dense", dense_logits), ("paged", paged_logits)):
                values, ids = logits[flip_index].float().topk(2)
                margin[label] = {
                    "top1_token_id": int(ids[0].item()),
                    "top1_logit": float(values[0].item()),
                    "top2_token_id": int(ids[1].item()),
                    "top2_logit": float(values[1].item()),
                    "margin": float((values[0] - values[1]).item()),
                }

        # The accepted path plus sibling/cousin examples exercises the decision
        # and the forbidden-cross-branch edges on the same frozen prefix.
        parents = [int(x) for x in tree.parent_indices.tolist()]
        siblings = next(
            ((i, j) for i in range(1, n) for j in range(i + 1, n)
             if parents[i] == parents[j]), None
        )
        cousins = next(
            ((i, j) for i in range(1, n) for j in range(i + 1, n)
             if int(tree.depth[i]) == int(tree.depth[j]) and parents[i] != parents[j]),
            None,
        )
        selected = set(int(x) for x in paged_path.tolist())
        selected.update(int(x) for x in dense_path.tolist())
        for pair in (siblings, cousins):
            if pair is not None:
                selected.update(pair)
        if flip_index is not None:
            selected.add(flip_index)
        selected = sorted(selected)
        branch_results = []
        branch_greedy = paged_greedy.clone()
        for node_id in selected:
            path = []
            current = node_id
            while current >= 0:
                path.append(current)
                current = parents[current]
            path.reverse()
            path_set = set(path)
            branch_blocks = self.block_manager.reserve_provisional(
                (n + self.block_size - 1) // self.block_size
            )
            try:
                block_ids = torch.tensor(branch_blocks, dtype=torch.long, device=slots.device)
                offsets_t = torch.arange(n, device=slots.device)
                branch_slots = block_ids[offsets_t // self.block_size] * self.block_size + offsets_t % self.block_size
                branch_logical = torch.cat((slots, branch_slots))
                branch_allowed = ancestor.clone()
                off_path = torch.tensor(
                    [i for i in range(n) if i not in path_set],
                    dtype=torch.long, device=slots.device,
                )
                branch_allowed[off_path] = False
                branch_allowed[off_path, off_path] = True
                branch_bias = torch.where(
                    branch_allowed,
                    torch.zeros((), dtype=torch.float32, device=slots.device),
                    torch.full((), float("-inf"), dtype=torch.float32, device=slots.device),
                )
                branch_tokens = tree.token_ids.clone()
                branch_tokens[off_path] = 0
                branch_logits, branch_taps, branch_final = self._target_forward_paged(
                    branch_tokens,
                    positions,
                    branch_slots, branch_logical, branch_bias,
                )
                branch_argmax = int(branch_logits[node_id].argmax().item())
                branch_greedy[node_id] = branch_argmax
                branch_results.append({
                    "node_index": node_id,
                    "ancestor_path": path,
                    "whole_tree_argmax": int(paged_greedy[node_id].item()),
                    "branchwise_argmax": branch_argmax,
                    "argmax_exact": branch_argmax == int(paged_greedy[node_id].item()),
                    "tap_diff": tensor_diff(branch_taps[node_id], paged_taps[node_id]),
                    "final_hidden_diff": tensor_diff(branch_final[node_id], paged_final_hidden[node_id]),
                    "logits_diff": tensor_diff(branch_logits[node_id], paged_logits[node_id]),
                    "logical_position": int(positions[node_id].item()),
                    "physical_slot": int(node_slots[node_id].item()),
                })
            finally:
                torch.cuda.current_stream(slots.device).synchronize()
                self.block_manager.release_provisional(branch_blocks)
        branch_path, branch_accepted, branch_correction = gpu_tree_accept(
            tree.token_ids, branch_greedy, tree.parent_indices, tree.depth,
            max_depth=self.tree_depth,
        )
        return {
            "prefix_len": prefix_len,
            "committed_token_ids": [int(x) for x in state.committed[0].tolist()],
            "target_feature_length": int(state.target_hidden.shape[1]),
            "draft_cache_length": int(self.drafter._fwd.cache.get_seq_length()),
            "root_token_id": int(tree.token_ids[0].item()),
            "tree_token_ids": [int(x) for x in tree.token_ids.tolist()],
            "tree_parent_indices": parents,
            "tree_depth": [int(x) for x in tree.depth.tolist()],
            "ancestor_matrix": ancestor.to(torch.uint8).cpu().tolist(),
            "logical_positions": [int(x) for x in positions.tolist()],
            "node_physical_slots": [int(x) for x in node_slots.tolist()],
            "physical_slots_disjoint_from_committed": not bool(
                torch.isin(node_slots, slots).any().item()
            ),
            "first_argmax_flip_node": flip_index,
            "first_argmax_flip_top2": margin,
            "dense_argmax_by_node": [int(x) for x in dense_greedy.tolist()],
            "paged_argmax_by_node": [int(x) for x in paged_greedy.tolist()],
            "singleton_root": singleton_root,
            "dense_decision": decision(dense_path, dense_accepted, dense_correction),
            "paged_decision": decision(paged_path, paged_accepted, paged_correction),
            "dense_paged_tap_diff": tensor_diff(dense_taps, paged_taps),
            "dense_paged_final_hidden_diff": tensor_diff(dense_final, paged_final_hidden),
            "dense_paged_logits_diff": tensor_diff(dense_logits, paged_logits),
            "branchwise_nodes": branch_results,
            "branchwise_decision": decision(branch_path, branch_accepted, branch_correction),
            "sibling_pair": list(siblings) if siblings else None,
            "cousin_pair": list(cousins) if cousins else None,
            "branchwise_semantic_decision_exact": (
                branch_path.tolist() == paged_path.tolist()
                and int(branch_correction.item()) == int(paged_correction.item())
            ),
        }

    @torch.inference_mode()
    def generate_target(self, prompt: str | list[int], max_new_tokens: int = 32) -> dict:
        """Pure greedy nano-vLLM target using the same correctness-first dense seam."""
        reset_context()
        ids = self._prompt_ids(prompt)
        device = ids.device
        plen = int(ids.shape[1])
        logits, cache, _ = self._target_forward(
            ids[0], torch.arange(plen, device=device), None,
            None, False,
        )
        token = logits[-1].argmax().view(1)
        output = [int(token.item())]
        while len(output) < max_new_tokens and output[-1] not in self.eos_token_ids:
            past_len = int(cache[0][0].shape[0])
            step_logits, new_kv, _ = self._target_forward(
                token, torch.tensor([past_len], device=device), cache,
                torch.ones((1, past_len + 1), dtype=torch.bool, device=device), False,
            )
            cache = [
                (torch.cat((old_k, nk), 0), torch.cat((old_v, nv), 0))
                for (old_k, old_v), (nk, nv) in zip(cache, new_kv)
            ]
            token = step_logits[-1].argmax().view(1)
            output.append(int(token.item()))
        return {"token_ids": output, "text": self.tokenizer.decode(output, skip_special_tokens=True)}

    def _prompt_ids(self, prompt: str | list[int]) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        return torch.tensor([prompt], dtype=torch.long, device="cuda")

    @torch.inference_mode()
    def generate(self, prompt: str | list[int], *, max_new_tokens: int = 32,
                 tree_backend: str = "dense",
                 qualification_rounds: tuple[int, ...] = (),
                 record_tree_layout: bool = False,
                 return_rounds: bool = True) -> dict[str, Any]:
        """Run a request with cleanup covering prefill, verify and commit failures."""
        if self._active_state is not None:
            raise RuntimeError("JetSpecRuntime already has an active request")
        try:
            return self._generate_request(
                prompt, max_new_tokens=max_new_tokens, tree_backend=tree_backend,
                qualification_rounds=qualification_rounds,
                record_tree_layout=record_tree_layout, return_rounds=return_rounds,
            )
        finally:
            try:
                if self._active_state is not None:
                    self._active_state.clear()
            finally:
                self._active_state = None
                self.drafter.reset_cache()
                reset_context()

    def _generate_request(self, prompt: str | list[int], *, max_new_tokens: int,
                          tree_backend: str, qualification_rounds: tuple[int, ...],
                          record_tree_layout: bool, return_rounds: bool) -> dict[str, Any]:
        if tree_backend not in ("dense", "paged"):
            raise ValueError("tree_backend must be 'dense' or 'paged'")
        if tree_backend == "paged" and (self.kv_pool is None or self.block_manager is None):
            raise RuntimeError("paged tree backend requires nano-vLLM KV pool and BlockManager")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        reset_context()
        self.drafter.reset_cache()
        input_ids = self._prompt_ids(prompt)
        device = input_ids.device
        prompt_len = int(input_ids.shape[1])
        torch.cuda.synchronize()
        start = time.perf_counter()

        prefill_logits, prompt_kv, prompt_hidden = self._target_forward(
            input_ids[0],
            torch.arange(prompt_len, device=device),
            None,
            None,
            True,
        )
        if prompt_hidden is None or prompt_hidden.shape[-1] != 20480:
            raise RuntimeError(f"invalid target feature shape: {None if prompt_hidden is None else prompt_hidden.shape}")
        first = prefill_logits[-1].argmax().view(1)
        committed = torch.cat((input_ids, first.view(1, 1)), dim=1)
        allocator_used_before = (
            len(self.block_manager.used_block_ids) if tree_backend == "paged" else 0
        )
        if tree_backend == "paged":
            state = PagedTargetState.from_prefill(
                committed,
                prompt_kv,
                prompt_hidden.unsqueeze(0),
                self.kv_pool,
                self.block_manager,
                self.block_size,
            )
        else:
            state = DenseTargetState(committed, prompt_kv, prompt_hidden.unsqueeze(0))
        self._active_state = state
        output_ids = [int(first.item())]
        rounds = []
        verify_events = []
        commit_events = []
        provisional_blocks_reserved = 0
        provisional_blocks_released_during_rounds = 0
        rejected_logical_slots = 0
        qualification_probes = []
        round_count = 0
        accept_lengths = []
        tree_sizes = []
        kv_copy_bytes = 0
        admission_latency = 0.0
        commit_host_latency = 0.0
        peak_used_blocks = len(state.owned_blocks) if tree_backend == "paged" else 0

        from jetspec.tree import build_ancestor_matrix, gpu_tree_accept

        while len(output_ids) < max_new_tokens and output_ids[-1] not in self.eos_token_ids:
            state.assert_round_invariant()
            draft_logits = self.drafter.propose_logits(
                state.committed,
                self.tree_depth,
                target_hidden=state.target_hidden,
            )
            tree = self.tree_algorithm.build(
                int(state.committed[0, -1]),
                draft_logits,
                self.tree_depth + 1,
                self.tree_width,
                self.tree_budget,
                device,
            )
            n = int(tree.num_nodes)
            past_len = state.cache_len
            ancestor = build_ancestor_matrix(tree).bool()
            positions = past_len + tree.depth.long()
            # Admission precedes verify: even the longest accepted path has a
            # canonical destination, independently of the scratch reservation.
            node_slots = None
            if tree_backend == "paged":
                admission_start = time.perf_counter()
                node_slots, logical_slots = state.reserve_tree(
                    n, max_path_length=min(n, self.tree_depth + 1)
                )
                admission_latency += time.perf_counter() - admission_start
                peak_used_blocks = max(
                    peak_used_blocks,
                    len(self.block_manager.used_block_ids) - allocator_used_before,
                )
            verify_start = torch.cuda.Event(enable_timing=True)
            verify_end = torch.cuda.Event(enable_timing=True)
            verify_start.record()
            if tree_backend == "paged":
                qq_bias = torch.where(
                    ancestor,
                    torch.zeros((), dtype=torch.float32, device=device),
                    torch.full((), float("-inf"), dtype=torch.float32, device=device),
                )
                target_logits, node_hidden, final_hidden = self._target_forward_paged(
                    tree.token_ids, positions, node_slots, logical_slots, qq_bias
                )
                provisional_kv = None
            else:
                allowed = torch.zeros((n, past_len + n), dtype=torch.bool, device=device)
                allowed[:, :past_len] = True
                allowed[:, past_len:] = ancestor
                target_logits, provisional_kv, node_hidden = self._target_forward(
                    tree.token_ids,
                    positions,
                    state.key_values,
                    allowed,
                    True,
                )
            verify_end.record()
            verify_events.append((verify_start, verify_end))
            if tree_backend == "paged" and round_count in qualification_rounds:
                qualification_probes.append({
                    "round_index": round_count,
                    **self._qualification_probe(
                        tree, state, ancestor, positions, node_slots,
                        target_logits, node_hidden, final_hidden,
                    ),
                })
            greedy = target_logits.argmax(dim=-1)
            path, accepted_len, correction = gpu_tree_accept(
                tree.token_ids,
                greedy,
                tree.parent_indices,
                tree.depth,
                max_depth=self.tree_depth,
            )
            if tree_backend == "paged" and not torch.equal(
                positions.index_select(0, path.long()),
                torch.arange(past_len, past_len + path.numel(), device=device),
            ):
                raise RuntimeError("accepted RoPE positions do not match canonical destination")
            commit_start = torch.cuda.Event(enable_timing=True)
            commit_end = torch.cuda.Event(enable_timing=True)
            commit_start.record()
            commit_host_start = time.perf_counter()
            accepted = tree.token_ids.index_select(0, path[1:])
            block = torch.cat((accepted, correction.view(1)))
            new_committed = torch.cat((state.committed, block.view(1, -1)), dim=1)
            if tree_backend == "paged":
                lifecycle = state.commit_tree_path(
                    node_slots, node_hidden.unsqueeze(0), path,
                    committed_tokens=new_committed,
                )
                provisional_blocks_reserved += lifecycle["reserved_blocks"]
                provisional_blocks_released_during_rounds += lifecycle["released_blocks"]
                rejected_logical_slots += lifecycle["rejected_logical_slots"]
                kv_copy_bytes += lifecycle["kv_copy_bytes"]
            else:
                state.commit_tree_path(provisional_kv, node_hidden.unsqueeze(0), path)
                state.committed = new_committed
            state.assert_round_invariant()
            commit_end.record()
            commit_host_latency += time.perf_counter() - commit_host_start
            commit_events.append((commit_start, commit_end))

            round_record = {
                "tree_size": n,
                "accepted_draft_length": int(accepted_len),
                "accepted_length_including_correction": int(accepted_len) + 1,
                "accepted_path_node_indices": [int(x) for x in path.tolist()],
                "accepted_path_token_ids_root_inclusive": [
                    int(x) for x in tree.token_ids.index_select(0, path).tolist()
                ],
                "target_argmax_token_by_tree_node": [int(x) for x in greedy.tolist()],
                "correction_token_id": int(correction.item()),
                "kv_copy_or_gather": True,
            }
            if tree_backend == "paged":
                round_record["kv_copy_bytes"] = lifecycle["kv_copy_bytes"]
                round_record["capacity"] = state.capacity_snapshot()
            if record_tree_layout:
                round_record["tree_token_ids"] = [int(x) for x in tree.token_ids.tolist()]
                round_record["tree_parent_indices"] = [int(x) for x in tree.parent_indices.tolist()]
                round_record["tree_depth"] = [int(x) for x in tree.depth.tolist()]
            if return_rounds:
                rounds.append(round_record)
            round_count += 1
            accept_lengths.append(int(accepted_len) + 1)
            tree_sizes.append(n)
            for value in block.tolist():
                output_ids.append(int(value))
                if int(value) in self.eos_token_ids:
                    break

        output_ids = output_ids[:max_new_tokens]
        torch.cuda.synchronize()
        latency = time.perf_counter() - start
        verify_latency = sum(start.elapsed_time(end) for start, end in verify_events) / 1000.0
        commit_latency = sum(start.elapsed_time(end) for start, end in commit_events) / 1000.0
        capacity_before_cleanup = state.capacity_snapshot() if tree_backend == "paged" else None
        final_invariant = {
            "committed_minus_one": int(state.committed.shape[1]) - 1,
            "target_kv_length": state.cache_len,
            "target_feature_length": int(state.target_hidden.shape[1]),
            "draft_cache_length": int(self.drafter._fwd.cache.get_seq_length()),
            "tree_backend": tree_backend,
            "pending_provisional_blocks": len(getattr(state, "pending_blocks", [])),
        }
        final_invariant["draft_pending_committed_feature_suffix"] = (
            final_invariant["target_feature_length"] - final_invariant["draft_cache_length"]
        )
        clean = (
            final_invariant["committed_minus_one"]
            == final_invariant["target_kv_length"]
            == final_invariant["target_feature_length"]
            and 0 <= final_invariant["draft_cache_length"] <= final_invariant["target_feature_length"]
            and final_invariant["pending_provisional_blocks"] == 0
        )
        # No generation state is reusable accidentally. A later call starts from empty
        # target state and reset Draft KV; rejected provisional tensors have no owner.
        blocks_released_cleanup = state.clear() if tree_backend == "paged" else 0
        if tree_backend == "dense":
            state.clear()
        self.drafter.reset_cache()
        self._active_state = None
        allocator_used_after = (
            len(self.block_manager.used_block_ids) if tree_backend == "paged" else 0
        )
        allocator_clean = allocator_used_after == allocator_used_before
        clean = clean and allocator_clean
        return {
            "token_ids": output_ids,
            "text": self.tokenizer.decode(output_ids, skip_special_tokens=True),
            "rounds": rounds,
            "target_verification_rounds": round_count,
            "accept_lengths": accept_lengths,
            "tree_sizes": tree_sizes,
            "latency_s": latency,
            "tree_backend": tree_backend,
            "qualification_probes": qualification_probes,
            "target_verification_latency_s": verify_latency,
            "kv_commit_reclaim_latency_s": commit_latency,
            "kv_copy_gather_rounds": round_count,
            "lifecycle_design": "canonical_committed_reusable_scratch" if tree_backend == "paged" else "dense",
            "kv_copy_bytes": kv_copy_bytes,
            "kv_copy_read_write_bytes": 2 * kv_copy_bytes,
            "capacity_before_cleanup": capacity_before_cleanup,
            "peak_used_blocks": peak_used_blocks,
            "peak_reserved_kv_slots": peak_used_blocks * self.block_size,
            "admission_latency_s": admission_latency,
            "commit_host_latency_s": commit_host_latency,
            "verify_latency_by_round_s": [s.elapsed_time(e) / 1000.0 for s, e in verify_events],
            "commit_latency_by_round_s": [s.elapsed_time(e) / 1000.0 for s, e in commit_events],
            "provisional_blocks_reserved": provisional_blocks_reserved,
            "provisional_blocks_released_during_rounds": provisional_blocks_released_during_rounds,
            "rejected_logical_slots": rejected_logical_slots,
            "blocks_released_cleanup": blocks_released_cleanup,
            "allocator_used_blocks_before": allocator_used_before,
            "allocator_used_blocks_after": allocator_used_after,
            "state_invariant_before_cleanup": final_invariant,
            "state_invariant_passed": clean,
            "stale_speculative_state_after_cleanup": (
                self._active_state is not None or not allocator_clean
            ),
        }
