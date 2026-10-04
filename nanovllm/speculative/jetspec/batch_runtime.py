"""Eager packed JetSpec steps with per-request state and runner-owned scratch.

This is the execution boundary used by the Continuous Batching scheduler. Requests
can be added, stepped in an arbitrary order, finished or cancelled between
steps. Target verification is one real packed forward; Draft proposals remain
per-request and share only immutable head weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
import time
from types import SimpleNamespace
from typing import Any

import torch

from nanovllm.speculative.jetspec.drafter import load_official_drafter
from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata
from nanovllm.speculative.jetspec.state import (
    BatchTreeTransaction, PagedTargetState, TreeScratchArena,
)
from nanovllm.utils.context import reset_context


@dataclass(eq=False)
class JetSpecRequest:
    request_id: str | int
    state: PagedTargetState
    drafter: object
    tree_budget: int
    max_new_tokens: int
    ignore_eos: bool
    prompt_length: int
    output_ids: list[int]
    created_at: float
    finished: bool = False
    cancelled: bool = False
    rounds: list[dict] = field(default_factory=list)
    result: dict | None = None
    preemptions: int = 0


class JetSpecBatchRuntime:
    """Independent Phase-3.1 runtime; does not call the legacy c1 runtime."""

    def __init__(self, target, tokenizer, draft_model: str, *, kv_pool,
                 block_manager, block_size: int = 256, tree_depth: int = 15,
                 tree_width: int = 7, max_tree_budget: int = 63,
                 max_verify_tokens: int = 4096, max_model_len: int = 4096,
                 head=None):
        if kv_pool is None or block_manager is None:
            raise ValueError("packed JetSpec requires a KV pool and allocator")
        if tree_depth != 15 or tree_width != 7:
            raise ValueError("current trained draft head requires depth=15, width=7")
        if max_tree_budget < 1 or max_verify_tokens < 1:
            raise ValueError("tree and packed query budgets must be positive")
        self.target, self.tokenizer = target, tokenizer
        self.kv_pool, self.block_manager = kv_pool, block_manager
        self.block_size, self.tree_depth, self.tree_width = block_size, tree_depth, tree_width
        self.max_tree_budget, self.max_verify_tokens = max_tree_budget, max_verify_tokens
        self.max_model_len = max_model_len
        if head is None:
            head, initial_drafter = load_official_drafter(draft_model, target, tree_depth)
            initial_drafter.reset_cache()
        self.head = head
        self.target_layer_ids = tuple(int(i) for i in head.target_layer_ids)
        if int(head.block_size) != tree_depth + 1 or self.target_layer_ids != (1, 9, 17, 25, 33):
            raise ValueError("unexpected trained draft head geometry/taps")
        from jetspec.tree import get_algorithm
        self.tree_algorithm = get_algorithm("accum_logp")
        self.arena = TreeScratchArena(kv_pool, block_manager, block_size)
        self.requests: dict[str | int, JetSpecRequest] = {}
        self._ids = count()
        self._active_transaction: BatchTreeTransaction | None = None
        self._closed = False
        self._reference_mode = False
        self.eos_token_ids = set()
        for value in (getattr(tokenizer, "eos_token_id", None),
                      getattr(getattr(target, "generation_config", None), "eos_token_id", None)):
            if value is not None:
                self.eos_token_ids.update(value if isinstance(value, (list, tuple, set)) else [value])

    def _check_idle(self) -> None:
        if self._closed:
            raise RuntimeError("JetSpec batch runtime is closed")
        if self._active_transaction is not None:
            raise RuntimeError("a packed transaction is in flight")

    def _prompt_ids(self, prompt) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not ids:
            raise ValueError("prompt must contain at least one token")
        return torch.tensor([ids], dtype=torch.long, device=self.kv_pool.device)

    def _new_drafter(self):
        from jetspec import DraftHeadTreeDrafter
        return DraftHeadTreeDrafter(
            self.head, target=self.target, block_size=self.head.block_size,
            target_layer_ids=self.target_layer_ids, draft_shift=False,
        )

    @torch.inference_mode()
    def create_request(self, prompt, *, max_new_tokens: int = 32, tree_budget: int = 63,
                       ignore_eos: bool = False, request_id: str | int | None = None) -> JetSpecRequest:
        self._check_idle()
        if max_new_tokens < 1 or not 1 <= tree_budget <= self.max_tree_budget:
            raise ValueError("invalid output limit or request tree budget")
        if request_id is None:
            request_id = next(self._ids)
            while request_id in self.requests:
                request_id = next(self._ids)
        if request_id in self.requests:
            raise ValueError("request ID is already live")
        created_at = time.perf_counter()
        ids = self._prompt_ids(prompt)
        prompt_length = int(ids.shape[1])
        # Verify includes speculative descendants beyond the output cap.
        if prompt_length + max_new_tokens + self.tree_depth > self.max_model_len:
            raise ValueError("prompt/output/tree lookahead exceed the configured model length")
        reset_context()
        hidden, prompt_kv, taps = self.target.model.forward_dense(
            ids[0], torch.arange(prompt_length, device=ids.device), None, None,
            self.target_layer_ids,
        )
        anchor = self.target.lm_head(hidden)[-1].argmax().reshape(1, 1)
        state = None
        try:
            state = PagedTargetState.from_prefill(
                torch.cat((ids, anchor), 1), prompt_kv, taps.unsqueeze(0),
                self.kv_pool, self.block_manager, self.block_size,
            )
            drafter = self._new_drafter()
            first = int(anchor.item())
            request = JetSpecRequest(
                request_id, state, drafter, int(tree_budget), int(max_new_tokens),
                bool(ignore_eos), prompt_length, [first], created_at,
                finished=max_new_tokens == 1 or (not ignore_eos and first in self.eos_token_ids),
            )
            state.assert_round_invariant()
            self.requests[request_id] = request
            return request
        except BaseException:
            if state is not None:
                state.clear()
            raise

    def capacity_snapshot(self) -> dict[str, int | float]:
        states = [r.state for r in self.requests.values()]
        committed = sum(len(s.owned_blocks) for s in states)
        pending = sum(len(s.pending_blocks) for s in states)
        scratch = len(self.arena.blocks)
        live = sum(s.cache_len for s in states)
        reserved = (committed + pending + scratch) * self.block_size
        return {"requests": len(states), "committed_blocks": committed,
                "pending_destination_blocks": pending, "scratch_blocks": scratch,
                "scratch_capacity_slots": self.arena.capacity,
                "live_kv_slots": live, "reserved_kv_slots": reserved,
                "allocator_used_blocks": len(self.block_manager.used_block_ids),
                "amplification": reserved / live if live else None}

    def estimate_prefill_capacity(self, num_cached_tokens: int, tree_budget: int,
                                  remaining_outputs: int) -> dict:
        """Read-only admission upper bound, including one decode's workspace.

        Recompute uses the saved committed prefix length, not original prompt
        length. The final admission is still the transaction's atomic reserve.
        """
        self._check_idle()
        if num_cached_tokens < 1 or not 1 <= tree_budget <= self.max_tree_budget or remaining_outputs < 0:
            raise ValueError("invalid prefill capacity estimate")
        b = self.block_size
        canonical = (num_cached_tokens + b - 1) // b
        path = min(tree_budget, self.tree_depth + 1, remaining_outputs)
        destination = (num_cached_tokens + path + b - 1) // b - canonical
        scratch = max(0, (tree_budget + b - 1) // b - len(self.arena.blocks)) if path else 0
        needed = canonical + destination + scratch
        return {"canonical_blocks": canonical, "destination_blocks": destination,
                "scratch_growth_blocks": scratch, "required_free_blocks": needed,
                "feasible": needed <= len(self.block_manager.free_block_ids)}

    def estimate_step_capacity(self, requests, tree_budgets=None) -> dict:
        """Conservative host plan; no pages/slots are reserved here."""
        self._check_idle()
        selected = self._selected(requests)
        budgets = self._step_budgets(selected, tree_budgets)
        b = self.block_size
        destinations = []
        for request, budget in zip(selected, budgets):
            path = min(budget, self.tree_depth + 1, request.max_new_tokens - len(request.output_ids))
            destinations.append((request.state.cache_len + path + b - 1) // b - len(request.state.owned_blocks))
        total = sum(budgets)
        scratch = max(0, (total + b - 1) // b - len(self.arena.blocks))
        needed = sum(destinations) + scratch
        return {"destination_blocks": sum(destinations), "destination_blocks_by_request": destinations,
                "scratch_growth_blocks": scratch, "required_free_blocks": needed,
                "total_query_tokens": total,
                "feasible": total <= self.max_verify_tokens and needed <= len(self.block_manager.free_block_ids)}

    def _step_budgets(self, selected, tree_budgets):
        budgets = [r.tree_budget for r in selected] if tree_budgets is None else list(tree_budgets)
        if len(budgets) != len(selected) or any(
                isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= r.tree_budget
                for r, n in zip(selected, budgets)):
            raise ValueError("one positive effective tree budget, within its request cap, is required per request")
        return budgets

    @torch.inference_mode()
    def suspend(self, request: JetSpecRequest) -> dict:
        """Recompute preemption: retain CPU tokens/anchor, release canonical KV.

        No output is generated and no runner-owned scratch is freed. The old
        runtime request is invalid after suspension; the scheduler owns its CPU
        snapshot until resume or cancellation.
        """
        self._check_idle()
        if self.requests.get(request.request_id) is not request or request.finished or request.cancelled:
            raise ValueError("only a live unfinished request can be suspended")
        request.state.assert_round_invariant()
        snapshot = {"request_id": request.request_id,
                    "committed_tokens": request.state.committed[0].tolist(),
                    "output_ids": list(request.output_ids), "tree_budget": request.tree_budget,
                    "max_new_tokens": request.max_new_tokens, "ignore_eos": request.ignore_eos,
                    "prompt_length": request.prompt_length, "created_at": request.created_at,
                    "rounds": list(request.rounds), "preemptions": request.preemptions + 1,
                    "state_invariant": {"kv_length": request.state.cache_len,
                        "feature_length": int(request.state.target_hidden.shape[1]),
                        "committed_minus_one": int(request.state.committed.shape[1]) - 1}}
        request.drafter.reset_cache()
        request.state.clear()
        del self.requests[request.request_id]
        return snapshot

    @torch.inference_mode()
    def resume(self, snapshot: dict) -> JetSpecRequest:
        """Rebuild historical KV/features without replacing/emitting the anchor."""
        self._check_idle()
        request_id = snapshot["request_id"]
        if request_id in self.requests:
            raise ValueError("request ID is already live")
        ids = self._prompt_ids(snapshot["committed_tokens"])
        prefix = ids[0, :-1]
        if prefix.numel() < 1 or len(snapshot["output_ids"]) >= snapshot["max_new_tokens"]:
            raise ValueError("invalid suspended request")
        reset_context()
        _, prompt_kv, taps = self.target.model.forward_dense(
            prefix, torch.arange(prefix.numel(), device=ids.device), None, None, self.target_layer_ids,
        )
        state = None
        try:
            state = PagedTargetState.from_prefill(ids, prompt_kv, taps.unsqueeze(0),
                self.kv_pool, self.block_manager, self.block_size)
            request = JetSpecRequest(request_id, state, self._new_drafter(),
                snapshot["tree_budget"], snapshot["max_new_tokens"], snapshot["ignore_eos"],
                snapshot["prompt_length"], list(snapshot["output_ids"]), snapshot["created_at"],
                rounds=list(snapshot["rounds"]), preemptions=snapshot["preemptions"])
            state.assert_round_invariant()
            self.requests[request_id] = request
            return request
        except BaseException:
            if state is not None:
                state.clear()
            raise

    def _selected(self, requests=None) -> list[JetSpecRequest]:
        selected = list(self.requests.values()) if requests is None else list(requests)
        if len({id(r) for r in selected}) != len(selected):
            raise ValueError("duplicate request in packed batch")
        for r in selected:
            if self.requests.get(r.request_id) is not r:
                raise ValueError("request does not belong to this runner")
        return [r for r in selected if not r.finished and not r.cancelled]

    def _verify_batch(self, requests, trees, transaction, metadata):
        transaction.arena.check_stream()
        tokens = torch.cat([t.token_ids for t in trees])
        positions = torch.cat([r.state.cache_len + t.depth.long() for r, t in zip(requests, trees)])
        hidden, taps = self.target.model.forward_packed_tree(
            tokens, positions, self.kv_pool, metadata, self.target_layer_ids,
        )
        return self.target.lm_head(hidden), taps

    @torch.inference_mode()
    def step(self, requests=None, *, tree_budgets=None) -> dict[str, Any]:
        """One packed verify/commit with a single publication boundary.

        Failures before commit preserve every prefix. After physical commit,
        even a reporting error retains the newly committed output/state; the
        caller must not replay that round as if it had rolled back.
        """
        self._check_idle()
        selected = self._selected(requests)
        budgets = self._step_budgets(selected, tree_budgets)
        if not selected:
            return {"request_ids": [], "total_query_tokens": 0, "capacity": self.capacity_snapshot()}
        if sum(budgets) > self.max_verify_tokens:
            raise ValueError("batch exceeds the packed verification token budget")
        from jetspec.tree import build_ancestor_matrix, gpu_tree_accept
        transaction = None
        publications = []

        def publish_outputs():
            # All list/dict preparation precedes commit. These assignments are
            # idempotent, so exception cleanup can finish a partial publication.
            for request, output_ids, finished, rounds in publications:
                request.output_ids = output_ids
                request.finished = finished
                request.rounds = rounds

        try:
            trees = []
            for r, budget in zip(selected, budgets):
                r.state.assert_round_invariant()
                if budget == 1 and not self._reference_mode:
                    # Pressure/output-tail fallback: genuine root verification,
                    # same packed attention/accept/commit, no useless Draft call.
                    trees.append(SimpleNamespace(
                        token_ids=r.state.committed[0, -1:].clone(),
                        depth=torch.zeros(1, dtype=torch.long, device=self.kv_pool.device),
                        parent_indices=torch.full((1,), -1, dtype=torch.long, device=self.kv_pool.device),
                        num_nodes=1, ancestor=torch.ones(1, 1, dtype=torch.bool, device=self.kv_pool.device),
                    ))
                    continue
                if self._reference_mode:
                    # One genuine AR root and T-1 isolated dummy queries. This
                    # is an explicit numerical comparator, never a serving path.
                    n = budget
                    trees.append(SimpleNamespace(
                        token_ids=torch.cat((r.state.committed[0, -1:], torch.zeros(
                            n - 1, dtype=torch.long, device=self.kv_pool.device))),
                        depth=torch.cat((torch.zeros(1, dtype=torch.long, device=self.kv_pool.device),
                                         torch.ones(n - 1, dtype=torch.long, device=self.kv_pool.device))),
                        num_nodes=n, ancestor=torch.eye(n, dtype=torch.bool, device=self.kv_pool.device),
                    ))
                    continue
                draft_logits = r.drafter.propose_logits(
                    r.state.committed, self.tree_depth, target_hidden=r.state.target_hidden,
                )
                trees.append(self.tree_algorithm.build(
                    int(r.state.committed[0, -1]), draft_logits,
                    self.tree_depth + 1, self.tree_width, budget, self.kv_pool.device,
                ))
            transaction = BatchTreeTransaction.admit(
                [r.state for r in selected], [int(t.num_nodes) for t in trees],
                [min(int(t.num_nodes), self.tree_depth + 1, r.max_new_tokens - len(r.output_ids))
                 for r, t in zip(selected, trees)], self.arena,
            )
            self._active_transaction = transaction
            metadata = PackedTreeMetadata.build(
                [r.state.cache_len for r in selected], [r.state.owned_blocks for r in selected],
                transaction.node_slots, [build_ancestor_matrix(t).bool() for t in trees],
                self.block_size, request_ids=[r.request_id for r in selected],
            )
            capacity_during_verify = self.capacity_snapshot()
            verify_start, verify_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            verify_start.record()
            logits, taps = self._verify_batch(selected, trees, transaction, metadata)
            verify_end.record()
            paths, next_tokens, features, request_records = [], [], [], []
            for i, (r, tree) in enumerate(zip(selected, trees)):
                lo, hi = metadata.query_offsets[i:i + 2]
                request_logits = logits[lo:hi]
                greedy = request_logits.argmax(-1)
                if self._reference_mode:
                    raw_path = torch.zeros(1, dtype=torch.long, device=greedy.device)
                    accepted_len, correction = 0, greedy[0]
                else:
                    raw_path, accepted_len, correction = gpu_tree_accept(
                        tree.token_ids, greedy, tree.parent_indices, tree.depth, max_depth=self.tree_depth,
                    )
                raw_outputs = torch.cat((tree.token_ids.index_select(0, raw_path[1:]), correction.reshape(1)))
                values = [int(x) for x in raw_outputs.tolist()]
                limit = min(len(values), r.max_new_tokens - len(r.output_ids))
                if not r.ignore_eos:
                    first_eos = next((j for j, token in enumerate(values[:limit]) if token in self.eos_token_ids), None)
                    if first_eos is not None:
                        limit = first_eos + 1
                path = raw_path[:limit]
                if not torch.equal(tree.depth.index_select(0, path.long()),
                                   torch.arange(path.numel(), device=path.device)):
                    raise RuntimeError("accepted RoPE positions do not match canonical tail")
                # Truncate the cached path with the emitted prefix, leaving its
                # last emitted token uncached even at EOS or max-token boundary.
                block = raw_outputs[:limit]
                paths.append(path)
                next_tokens.append(torch.cat((r.state.committed, block.reshape(1, -1)), 1))
                features.append(taps[lo:hi].unsqueeze(0))
                record = {
                    "request_id": r.request_id, "tree_size": int(tree.num_nodes),
                    "effective_tree_budget": budgets[i],
                    "accepted_draft_length": int(accepted_len),
                    "committed_path_indices": [int(x) for x in path.tolist()],
                    "raw_accepted_path_indices": [int(x) for x in raw_path.tolist()],
                    "verification_correction_token_id": int(correction.item()),
                    "output_block": values[:limit],
                    "target_argmax_by_node": [int(x) for x in greedy.tolist()],
                    "kv_length": r.state.cache_len + int(path.numel()),
                    "feature_length": r.state.cache_len + int(path.numel()),
                    "committed_minus_one": int(next_tokens[-1].shape[1]) - 1,
                }
                request_records.append(record)
                next_output_ids = r.output_ids + values[:limit]
                finished = (len(next_output_ids) >= r.max_new_tokens or
                            (not r.ignore_eos and next_output_ids[-1] in self.eos_token_ids))
                publications.append((r, next_output_ids, finished, r.rounds + [record]))
            commit_start, commit_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            commit_start.record()
            lifecycle = transaction.commit(features, paths, next_tokens)
            publish_outputs()
            commit_end.record()
            self._active_transaction = None
            for r in selected:
                r.state.assert_round_invariant()
            return {"request_ids": [r.request_id for r in selected],
                    "node_counts": [int(t.num_nodes) for t in trees],
                    "total_query_tokens": int(logits.shape[0]),
                    "cu_seqlens_q": list(metadata.query_offsets),
                    "requests": request_records, "lifecycle": lifecycle,
                    "capacity_during_verify": capacity_during_verify,
                    "capacity": self.capacity_snapshot(),
                    "_verify_events": (verify_start, verify_end),
                    "_commit_events": (commit_start, commit_end)}
        except BaseException:
            if transaction is not None:
                if getattr(transaction, "committed", not transaction.active):
                    publish_outputs()
                    transaction.abort()  # committed transactions only finish guard cleanup
                else:
                    transaction.abort()
            self._active_transaction = None
            for r in selected:
                r.drafter.reset_cache()
            raise

    @torch.inference_mode()
    def finish(self, request: JetSpecRequest) -> dict:
        self._check_idle()
        if request.result is not None:
            return request.result
        if self.requests.get(request.request_id) is not request:
            raise ValueError("request does not belong to this runner")
        if not request.finished and not request.cancelled:
            raise RuntimeError("unfinished requests must be stepped or explicitly cancelled")
        request.state.assert_round_invariant()
        invariant = {"kv_length": request.state.cache_len,
                     "feature_length": int(request.state.target_hidden.shape[1]),
                     "committed_minus_one": int(request.state.committed.shape[1]) - 1}
        # Prepare all fallible result construction before releasing ownership.
        result = {"request_id": request.request_id, "token_ids": list(request.output_ids),
                  "text": self.tokenizer.decode(request.output_ids, skip_special_tokens=True),
                  "tree_budget": request.tree_budget, "rounds": list(request.rounds),
                  "cancelled": request.cancelled, "state_invariant": invariant,
                  "preemptions": request.preemptions,
                  "latency_s": time.perf_counter() - request.created_at}
        released = request.state.clear()
        del self.requests[request.request_id]
        result["blocks_released"] = released
        request.result = result
        # Failure here must not put a cleared state back into the live registry.
        request.drafter.reset_cache()
        return request.result

    def cancel(self, request: JetSpecRequest) -> dict:
        self._check_idle()
        if request.result is not None:
            return request.result
        if self.requests.get(request.request_id) is not request:
            raise ValueError("request does not belong to this runner")
        request.cancelled = True
        return self.finish(request)

    def release_idle_scratch(self) -> int:
        self._check_idle()
        if self.requests:
            raise RuntimeError("cannot release runner scratch while requests are live")
        return self.arena.clear()

    @torch.inference_mode()
    def generate_batch(self, prompts, *, max_new_tokens=32, tree_budgets=63,
                       ignore_eos: bool = False, return_rounds: bool = True) -> dict:
        self._check_idle()
        if self.requests:
            raise RuntimeError("generate_batch requires no existing live requests; use step for admission")
        prompts = list(prompts)
        if not prompts:
            raise ValueError("at least one prompt is required")
        limits = [max_new_tokens] * len(prompts) if isinstance(max_new_tokens, int) else list(max_new_tokens)
        budgets = [tree_budgets] * len(prompts) if isinstance(tree_budgets, int) else list(tree_budgets)
        if len(limits) != len(prompts) or len(budgets) != len(prompts):
            raise ValueError("one output limit and tree budget are required per prompt")
        torch.cuda.synchronize(self.kv_pool.device)
        start = time.perf_counter()
        before = len(self.block_manager.used_block_ids)
        created, rounds, completed = [], [], {}
        peak_slots = self.arena.capacity
        try:
            for prompt, limit, budget in zip(prompts, limits, budgets):
                created.append(self.create_request(prompt, max_new_tokens=limit, tree_budget=budget,
                                                   ignore_eos=ignore_eos))
            peak_slots = max(peak_slots, self.capacity_snapshot()["reserved_kv_slots"])
            while self.requests:
                for r in created:
                    if r.result is None and r.finished:
                        completed[r.request_id] = self.finish(r)
                if not self.requests:
                    break
                record = self.step()
                peak_slots = max(peak_slots, record["capacity_during_verify"]["reserved_kv_slots"])
                rounds.append(record)
            torch.cuda.synchronize(self.kv_pool.device)
            latency = time.perf_counter() - start
            for record in rounds:
                vs, ve = record.pop("_verify_events")
                cs, ce = record.pop("_commit_events")
                record["verify_latency_s"] = vs.elapsed_time(ve) / 1000.0
                record["commit_latency_s"] = cs.elapsed_time(ce) / 1000.0
            results = [completed[r.request_id] for r in created]
            if not return_rounds:
                for result in results:
                    result["rounds"] = []
            total_tokens = sum(len(r["token_ids"]) for r in results)
            return {"requests": results, "rounds": rounds if return_rounds else [],
                    "packed_verification_rounds": len(rounds),
                    "latency_s": latency, "total_output_tokens": total_tokens,
                    "tokens_per_second": total_tokens / latency,
                    "target_verification_latency_s": sum(r["verify_latency_s"] for r in rounds),
                    "kv_commit_latency_s": sum(r["commit_latency_s"] for r in rounds),
                    "kv_copy_bytes": sum(r["lifecycle"]["kv_copy_bytes"] for r in rounds),
                    "peak_reserved_kv_slots": peak_slots,
                    "allocator_used_blocks_before": before,
                    "allocator_used_blocks_after": len(self.block_manager.used_block_ids),
                    "runner_scratch_blocks": len(self.arena.blocks),
                    "request_cleanup_passed": not self.requests,
                    "numerical_path": "packed_fp32_ragged",
                    "reference_mode": self._reference_mode,
                    "scratch_owner": "runner"}
        finally:
            if self._active_transaction is not None:
                self._active_transaction.abort()
                self._active_transaction = None
            for r in created:
                if self.requests.get(r.request_id) is r:
                    self.cancel(r)
            reset_context()

    @torch.inference_mode()
    def generate_target_batch(self, prompts, *, max_new_tokens=32, tree_budgets=63,
                              ignore_eos: bool = False, return_rounds: bool = True) -> dict:
        """Padded-root AR comparator using the same packed FP32 backend.

        Rows per live request match its tree budget. Request-removal schedules
        can differ from speculative decoding; this is not a claim of matching
        GEMM shapes at every closed-loop token. Frozen same-Q isolation is the
        separate semantic correctness gate.
        """
        self._check_idle()
        self._reference_mode = True
        try:
            return self.generate_batch(prompts, max_new_tokens=max_new_tokens,
                                       tree_budgets=tree_budgets, ignore_eos=ignore_eos,
                                       return_rounds=return_rounds)
        finally:
            self._reference_mode = False

    def close(self) -> None:
        if self._closed:
            return
        if self._active_transaction is not None:
            self._active_transaction.abort()
            self._active_transaction = None
        for r in list(self.requests.values()):
            self.cancel(r)
        self.arena.clear()
        self._closed = True
        reset_context()
