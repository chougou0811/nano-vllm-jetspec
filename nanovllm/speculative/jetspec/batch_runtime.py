"""Eager packed JetSpec steps with per-request state and runner-owned scratch.

This is the execution boundary used by the Continuous Batching scheduler. Requests
can be added, stepped in an arbitrary order, finished or cancelled between
steps. Target verification is one real packed forward; serving mode also groups
compatible Draft proposals while preserving compact, request-owned Draft KV.
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
        self.prefills = {}
        self._ids = count()
        self._active_transaction: BatchTreeTransaction | None = None
        self._closed = False
        self._reference_mode = False
        self._lightweight = False
        self._batched_draft_enabled = False
        self._feature_storage = False
        self._attention_backend = "sdpa"
        self._batch_proposer = None
        self.eos_token_ids = set()
        for value in (getattr(tokenizer, "eos_token_id", None),
                      getattr(getattr(target, "generation_config", None), "eos_token_id", None)):
            if value is not None:
                self.eos_token_ids.update(value if isinstance(value, (list, tuple, set)) else [value])

    def configure_optimizations(self, *, lightweight=False, batched_draft=False,
                                feature_storage=False, attention_backend="sdpa"):
        """Select independently measurable serving paths at an idle boundary."""
        self._check_idle()
        if self.requests or getattr(self, "prefills", {}):
            raise RuntimeError("optimization policy cannot change with live requests")
        if attention_backend not in ("sdpa", "flash_attn"):
            raise ValueError("JetSpec attention backend must be 'sdpa' or 'flash_attn'")
        if attention_backend == "flash_attn":
            if not batched_draft:
                raise ValueError("FlashAttention Draft requires the batched serving adapter")
            from nanovllm.speculative.jetspec.flash_prefill import require_flash_attention
            require_flash_attention()
        self._lightweight = bool(lightweight)
        self._batched_draft_enabled = bool(batched_draft)
        self._feature_storage = bool(feature_storage)
        self._attention_backend = attention_backend
        self._batch_proposer = None

    def _configure_state(self, state):
        state.validate_device = not getattr(self, "_lightweight", False)
        if getattr(self, "_feature_storage", False) and state._feature_storage is None:
            state.enable_feature_storage(max_capacity=self.max_model_len)

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
    def begin_prefill(self, prompt=None, *, max_new_tokens=32, tree_budget=63,
                      ignore_eos=False, request_id=None, snapshot=None):
        """Register page-empty PREFILL/RECOMPUTE state, separate from READY.

        Resume rebuilds all historical rows except the saved uncached anchor;
        it never predicts or emits another anchor. Call prefill_step repeatedly
        with a bounded query budget, then use the returned READY request.
        """
        from nanovllm.speculative.jetspec.prefill import PrefillContext
        self._check_idle()
        if not hasattr(self, "prefills"):
            self.prefills = {}
        saved = None
        created_at = time.perf_counter()
        if snapshot is not None:
            if prompt is not None or (request_id is not None and request_id != snapshot["request_id"]):
                raise ValueError("resume accepts a snapshot, not a replacement prompt/request ID")
            saved = dict(snapshot)
            saved["committed_tokens"] = list(snapshot["committed_tokens"])
            saved["output_ids"] = list(snapshot["output_ids"])
            saved["rounds"] = list(snapshot["rounds"])
            request_id, tree_budget = saved["request_id"], saved["tree_budget"]
            max_new_tokens, ignore_eos = saved["max_new_tokens"], saved["ignore_eos"]
            created_at = saved["created_at"]
            if (len(saved["committed_tokens"]) < 2 or not saved["output_ids"] or
                    len(saved["output_ids"]) >= max_new_tokens or
                    saved["prompt_length"] < 1 or
                    len(saved["committed_tokens"]) != saved["prompt_length"] + len(saved["output_ids"]) or
                    saved["committed_tokens"][-len(saved["output_ids"]):] != saved["output_ids"]):
                raise ValueError("invalid suspended request tokens/output anchor")
            ids = self._prompt_ids(saved["committed_tokens"][:-1])
            prompt_length = saved["prompt_length"]
        else:
            ids = self._prompt_ids(prompt)
            prompt_length = int(ids.shape[1])
        if max_new_tokens < 1 or not 1 <= tree_budget <= self.max_tree_budget:
            raise ValueError("invalid output limit or request tree budget")
        # Match original admission's conservative lookahead bound. Recompute
        # tokens exclude its saved anchor; do not accidentally gain one token
        # of model-length allowance by validating only prefix+remaining.
        if prompt_length + max_new_tokens + self.tree_depth > self.max_model_len:
            raise ValueError("prompt/output/tree lookahead exceed the configured model length")
        if request_id is None:
            request_id = next(self._ids)
            while request_id in self.requests or request_id in self.prefills:
                request_id = next(self._ids)
        if request_id in self.requests or request_id in self.prefills:
            raise ValueError("request ID is already live")
        context = PrefillContext(request_id, ids, self.kv_pool, self.block_manager,
            self.block_size, int(tree_budget), int(max_new_tokens), bool(ignore_eos),
            prompt_length, created_at, snapshot=saved)
        # Token construction can occur on a different stream from chunk one.
        context.record_ready()
        self.prefills[request_id] = context
        return context

    def _prefill_owned(self, context):
        self._check_idle()
        if getattr(self, "prefills", {}).get(context.request_id) is not context:
            raise ValueError("prefill context does not belong to this runner")

    def estimate_prefill_chunk_capacity(self, context, num_tokens):
        self._prefill_owned(context)
        return context.capacity(num_tokens)

    @torch.inference_mode()
    def prefill_step(self, context, num_tokens):
        """Append one chunk; publish exactly once only when the prefix is ready."""
        from nanovllm.speculative.jetspec.prefill import forward_chunk
        self._prefill_owned(context)
        count = context.chunk_length(num_tokens)
        if not context.capacity(count)["feasible"]:
            # Expected allocator backpressure has not queued any writes and
            # preserves the existing partial prefix for a later retry.
            raise RuntimeError("insufficient KV blocks for prefill chunk")
        try:
            reset_context()
            hidden = forward_chunk(self.target, context, count, self.target_layer_ids,
                **({"attention_backend": self._attention_backend}
                   if getattr(self, "_attention_backend", "sdpa") != "sdpa" else {}))
            if context.remaining_tokens:
                return None
            context.begin_writes()
            if context.snapshot is None:
                anchor = self.target.lm_head(hidden[-1:])[-1].argmax().reshape(1, 1)
                committed = torch.cat((context.tokens, anchor), 1)
                output_ids = [int(anchor.item())]
                rounds, preemptions = [], 0
            else:
                saved = context.snapshot
                committed = self._prompt_ids(saved["committed_tokens"])
                output_ids = list(saved["output_ids"])
                rounds, preemptions = list(saved["rounds"]), saved["preemptions"]
            scratch = TreeScratchArena(self.kv_pool, self.block_manager, self.block_size)
            scratch._retired = context._ready
            state = PagedTargetState(committed, context.target_hidden, self.kv_pool,
                self.block_manager, self.block_size, context.logical_slots,
                list(context.owned_blocks), [], scratch=scratch)
            # Keep the prefill capacity backing; decode appends without cat.
            state._feature_storage = context.feature_storage
            state._feature_max_capacity = self.max_model_len
            state._feature_append_bytes = sum(r["feature_append_bytes"] for r in context.chunk_records)
            self._configure_state(state)
            request = JetSpecRequest(context.request_id, state, self._new_drafter(),
                context.tree_budget, context.max_new_tokens, context.ignore_eos,
                context.prompt_length, output_ids, context.created_at,
                finished=(len(output_ids) >= context.max_new_tokens or
                    (not context.ignore_eos and output_ids[-1] in self.eos_token_ids)),
                rounds=rounds, preemptions=preemptions)
            state.assert_round_invariant()
            # Promotion itself creates committed/slot metadata and may enqueue
            # feature operations. Fence ALL of it, not just the last chunk/head.
            # Keep _writing_stream live until here so exceptions also fence any
            # promotion work not covered by the previous chunk's ready event.
            context.record_ready()
            scratch._retired = context._ready
            self.requests[request.request_id] = request
            del self.prefills[context.request_id]
            context.owned_blocks = []
            context.feature_storage = None
            context.promoted = True
            return request
        except BaseException as exception:
            context.error = f"{type(exception).__name__}: {exception}"
            try:
                self.cancel_prefill(context)
            except BaseException as cleanup:
                exception.add_note(f"partial prefill cleanup failed: {cleanup}")
            raise

    def cancel_prefill(self, context):
        self._check_idle()
        if context.promoted or context.cancelled:
            return 0
        self._prefill_owned(context)
        released = context.clear()
        del self.prefills[context.request_id]
        return released

    @torch.inference_mode()
    def create_request(self, prompt, *, max_new_tokens: int = 32, tree_budget: int = 63,
                       ignore_eos: bool = False, request_id: str | int | None = None) -> JetSpecRequest:
        self._check_idle()
        if max_new_tokens < 1 or not 1 <= tree_budget <= self.max_tree_budget:
            raise ValueError("invalid output limit or request tree budget")
        if request_id is None:
            request_id = next(self._ids)
            while request_id in self.requests or request_id in getattr(self, "prefills", {}):
                request_id = next(self._ids)
        if request_id in self.requests or request_id in getattr(self, "prefills", {}):
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
            **({"attention_backend": self._attention_backend}
               if getattr(self, "_attention_backend", "sdpa") != "sdpa" else {}),
        )
        # Only the final prompt row predicts the first output. The debug path
        # retains the qualified full-prefill GEMM shape for numerical controls.
        head_input = hidden[-1:] if getattr(self, "_lightweight", False) else hidden
        anchor = self.target.lm_head(head_input)[-1].argmax().reshape(1, 1)
        state = None
        try:
            state = PagedTargetState.from_prefill(
                torch.cat((ids, anchor), 1), prompt_kv, taps.unsqueeze(0),
                self.kv_pool, self.block_manager, self.block_size,
            )
            self._configure_state(state)
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
        partials = list(getattr(self, "prefills", {}).values())
        committed = sum(len(s.owned_blocks) for s in states)
        partial_blocks = sum(len(c.owned_blocks) for c in partials)
        pending = sum(len(s.pending_blocks) for s in states)
        scratch = len(self.arena.blocks)
        live = sum(s.cache_len for s in states) + sum(c.processed_tokens for c in partials)
        reserved = (committed + partial_blocks + pending + scratch) * self.block_size
        features = [s.feature_storage_snapshot() for s in states]
        from nanovllm.speculative.jetspec.batched_draft import BatchedDraftProposer
        return {"requests": len(states), "committed_blocks": committed,
                "prefill_requests": len(partials), "prefill_blocks": partial_blocks,
                "prefill_processed_tokens": sum(c.processed_tokens for c in partials),
                "pending_destination_blocks": pending, "scratch_blocks": scratch,
                "scratch_capacity_slots": self.arena.capacity,
                "live_kv_slots": live, "reserved_kv_slots": reserved,
                "allocator_used_blocks": len(self.block_manager.used_block_ids),
                "amplification": reserved / live if live else None,
                "target_feature_reserved_bytes": sum(s["feature_reserved_bytes"] for s in features) + sum(
                    c.feature_storage.numel() * c.feature_storage.element_size() for c in partials if c.feature_storage is not None),
                "target_feature_live_bytes": sum(s.target_hidden.numel() * s.target_hidden.element_size() for s in states) + sum(
                    c.target_hidden.numel() * c.target_hidden.element_size() for c in partials if c.target_hidden is not None),
                "draft_cache_bytes": BatchedDraftProposer._cache_bytes(self.requests.values()),
                "memory_model": "admission budgets Target KV pages; auxiliary bytes are observed, not a full CUDA admission model"}

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
        if request_id in self.requests or request_id in getattr(self, "prefills", {}):
            raise ValueError("request ID is already live")
        ids = self._prompt_ids(snapshot["committed_tokens"])
        prefix = ids[0, :-1]
        if prefix.numel() < 1 or len(snapshot["output_ids"]) >= snapshot["max_new_tokens"]:
            raise ValueError("invalid suspended request")
        reset_context()
        _, prompt_kv, taps = self.target.model.forward_dense(
            prefix, torch.arange(prefix.numel(), device=ids.device), None, None, self.target_layer_ids,
            **({"attention_backend": self._attention_backend}
               if getattr(self, "_attention_backend", "sdpa") != "sdpa" else {}),
        )
        state = None
        try:
            state = PagedTargetState.from_prefill(ids, prompt_kv, taps.unsqueeze(0),
                self.kv_pool, self.block_manager, self.block_size)
            self._configure_state(state)
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

    def _propose_drafts(self, requests):
        if not requests:
            return []
        if getattr(self, "_batched_draft_enabled", False) and hasattr(self, "head"):
            if self._batch_proposer is None:
                if getattr(self, "_attention_backend", "sdpa") == "flash_attn":
                    from nanovllm.speculative.jetspec.flash_draft import FlashDraftProposer
                    self._batch_proposer = FlashDraftProposer(self.head, self.target)
                else:
                    from nanovllm.speculative.jetspec.batched_draft import BatchedDraftProposer
                    self._batch_proposer = BatchedDraftProposer(self.head, self.target)
            return self._batch_proposer.propose(requests, depth=self.tree_depth)
        return [r.drafter.propose_logits(r.state.committed, self.tree_depth,
                                        target_hidden=r.state.target_hidden) for r in requests]

    def _build_trees(self, selected, budgets):
        active = [r for r, budget in zip(selected, budgets) if budget > 1]
        proposals = iter(self._propose_drafts(active)) if not self._reference_mode else iter(())
        aligned = [next(proposals) if budget > 1 and not self._reference_mode else None for budget in budgets]
        if (getattr(self, "_lightweight", False) and not self._reference_mode and
                getattr(self.tree_algorithm, "name", None) == "accum_logp"):
            from nanovllm.speculative.jetspec.serving_ops import build_trees
            return build_trees([r.output_ids[-1] for r in selected], aligned, budgets,
                               self.tree_depth, self.tree_width, self.kv_pool.device)
        trees = []
        for r, budget, draft_logits in zip(selected, budgets, aligned):
            if self._reference_mode:
                trees.append(SimpleNamespace(
                    token_ids=torch.cat((r.state.committed[0, -1:], torch.zeros(
                        budget - 1, dtype=torch.long, device=self.kv_pool.device))),
                    depth=torch.cat((torch.zeros(1, dtype=torch.long, device=self.kv_pool.device),
                                     torch.ones(budget - 1, dtype=torch.long, device=self.kv_pool.device))),
                    num_nodes=budget, ancestor=torch.eye(budget, dtype=torch.bool, device=self.kv_pool.device)))
            elif budget == 1:
                trees.append(SimpleNamespace(
                    token_ids=r.state.committed[0, -1:].clone(),
                    depth=torch.zeros(1, dtype=torch.long, device=self.kv_pool.device),
                    parent_indices=torch.full((1,), -1, dtype=torch.long, device=self.kv_pool.device),
                    num_nodes=1, ancestor=torch.ones(1, 1, dtype=torch.bool, device=self.kv_pool.device)))
            else:
                trees.append(self.tree_algorithm.build(int(r.state.committed[0, -1]), draft_logits,
                    self.tree_depth + 1, self.tree_width, budget, self.kv_pool.device))
        return trees

    def _accept_batch(self, logits, trees, metadata):
        from nanovllm.speculative.jetspec.serving_ops import accept_batch
        return accept_batch(logits, trees, metadata.query_offsets, self.tree_depth)

    @torch.inference_mode()
    def step(self, requests=None, *, tree_budgets=None, record_timing=None) -> dict[str, Any]:
        """One packed verify/commit with a single publication boundary.

        Failures before commit preserve every prefix. After physical commit,
        even a reporting error retains the newly committed output/state; the
        caller must not replay that round as if it had rolled back.
        Timing events default off in lightweight serving; diagnostic wrappers
        may explicitly request them without enabling full node records.
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
        lightweight = getattr(self, "_lightweight", False) and not self._reference_mode
        timed = not lightweight if record_timing is None else bool(record_timing)

        def publish_outputs():
            # All list/dict preparation precedes commit. These assignments are
            # idempotent, so exception cleanup can finish a partial publication.
            for request, output_ids, finished, rounds in publications:
                request.output_ids = output_ids
                request.finished = finished
                request.rounds = rounds

        try:
            for r in selected:
                r.state.assert_round_invariant()
            trees = self._build_trees(selected, budgets)
            transaction = BatchTreeTransaction.admit(
                [r.state for r in selected], [int(t.num_nodes) for t in trees],
                [min(int(t.num_nodes), self.tree_depth + 1, r.max_new_tokens - len(r.output_ids))
                 for r, t in zip(selected, trees)], self.arena,
            )
            self._active_transaction = transaction
            if lightweight and all(hasattr(t, "host_ancestor") for t in trees):
                metadata = PackedTreeMetadata.from_host_trees(
                    [r.state.cache_len for r in selected], [r.state.owned_blocks for r in selected],
                    trees, self.arena.blocks, self.block_size, device=self.kv_pool.device,
                    request_ids=[r.request_id for r in selected])
            else:
                metadata = PackedTreeMetadata.build(
                    [r.state.cache_len for r in selected], [r.state.owned_blocks for r in selected],
                    transaction.node_slots, [build_ancestor_matrix(t).bool() for t in trees],
                    self.block_size, request_ids=[r.request_id for r in selected])
            capacity_during_verify = self.capacity_snapshot()
            if timed:
                verify_start, verify_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                verify_start.record()
            logits, taps = self._verify_batch(selected, trees, transaction, metadata)
            if timed:
                verify_end.record()
            accepted = self._accept_batch(logits, trees, metadata) if lightweight else None
            paths, next_tokens, features, request_records = [], [], [], []
            host_paths = []
            for i, (r, tree) in enumerate(zip(selected, trees)):
                lo, hi = metadata.query_offsets[i:i + 2]
                if lightweight:
                    outcome = accepted[i]
                    values = outcome["outputs"]
                    raw_path_host = outcome["path"]
                    accepted_len = outcome["accepted_length"]
                    correction_id = outcome["correction"]
                else:
                    greedy = logits[lo:hi].argmax(-1)
                    if self._reference_mode:
                        raw_path = torch.zeros(1, dtype=torch.long, device=greedy.device)
                        accepted_len, correction = 0, greedy[0]
                    else:
                        raw_path, accepted_len, correction = gpu_tree_accept(
                            tree.token_ids, greedy, tree.parent_indices, tree.depth, max_depth=self.tree_depth)
                    raw_outputs = torch.cat((tree.token_ids.index_select(0, raw_path[1:]), correction.reshape(1)))
                    values = [int(x) for x in raw_outputs.tolist()]
                    raw_path_host = [int(x) for x in raw_path.tolist()]
                    correction_id = int(correction.item())
                limit = min(len(values), r.max_new_tokens - len(r.output_ids))
                if not r.ignore_eos:
                    first_eos = next((j for j, token in enumerate(values[:limit]) if token in self.eos_token_ids), None)
                    if first_eos is not None:
                        limit = first_eos + 1
                path_host = raw_path_host[:limit]
                if lightweight:
                    path = torch.tensor(path_host, dtype=torch.long)
                    block = torch.tensor(values[:limit], dtype=torch.long, device=self.kv_pool.device)
                else:
                    path = raw_path[:limit]
                    if not torch.equal(tree.depth.index_select(0, path.long()),
                                       torch.arange(path.numel(), device=path.device)):
                        raise RuntimeError("accepted RoPE positions do not match canonical tail")
                    block = raw_outputs[:limit]
                # Truncate the cached path with the emitted prefix, leaving its
                # last emitted token uncached even at EOS or max-token boundary.
                paths.append(path)
                host_paths.append(path_host)
                next_tokens.append(torch.cat((r.state.committed, block.reshape(1, -1)), 1))
                features.append(taps[lo:hi].unsqueeze(0))
                record = {
                    "request_id": r.request_id, "tree_size": int(tree.num_nodes),
                    "effective_tree_budget": budgets[i],
                    "accepted_draft_length": int(accepted_len),
                    "committed_path_indices": path_host,
                    "raw_accepted_path_indices": raw_path_host,
                    "verification_correction_token_id": correction_id,
                    "output_block": values[:limit],
                    "kv_length": r.state.cache_len + int(path.numel()),
                    "feature_length": r.state.cache_len + int(path.numel()),
                    "committed_minus_one": int(next_tokens[-1].shape[1]) - 1,
                }
                if not lightweight:
                    record["target_argmax_by_node"] = [int(x) for x in greedy.tolist()]
                request_records.append(record)
                next_output_ids = r.output_ids + values[:limit]
                finished = (len(next_output_ids) >= r.max_new_tokens or
                            (not r.ignore_eos and next_output_ids[-1] in self.eos_token_ids))
                publications.append((r, next_output_ids, finished, r.rounds + [record]))
            if timed:
                commit_start, commit_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                commit_start.record()
            if lightweight:
                lifecycle = transaction.commit(features, paths, next_tokens, accepted_paths_host=host_paths)
            else:
                lifecycle = transaction.commit(features, paths, next_tokens)
            publish_outputs()
            if timed:
                commit_end.record()
            self._active_transaction = None
            for r in selected:
                r.state.assert_round_invariant()
            result = {"request_ids": [r.request_id for r in selected],
                    "node_counts": [int(t.num_nodes) for t in trees],
                    "total_query_tokens": int(logits.shape[0]),
                    "cu_seqlens_q": list(metadata.query_offsets),
                    "requests": request_records, "lifecycle": lifecycle,
                    "capacity_during_verify": capacity_during_verify,
                    "capacity": self.capacity_snapshot()}
            if timed:
                result.update(_verify_events=(verify_start, verify_end), _commit_events=(commit_start, commit_end))
            return result
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
        # Partial prefixes own canonical pages, never runner scratch. Permit
        # releasing an oversized idle arena to relieve incremental admission.
        if self.requests:
            raise RuntimeError("cannot release runner scratch while requests are live")
        return self.arena.clear()

    @torch.inference_mode()
    def generate_batch(self, prompts, *, max_new_tokens=32, tree_budgets=63,
                       ignore_eos: bool = False, return_rounds: bool = True) -> dict:
        self._check_idle()
        if self.requests or getattr(self, "prefills", {}):
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
                record = self.step(record_timing=True)
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
        for context in list(getattr(self, "prefills", {}).values()):
            self.cancel_prefill(context)
        for r in list(self.requests.values()):
            self.cancel(r)
        self.arena.clear()
        self._closed = True
        reset_context()
