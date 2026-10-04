"""Synchronous Continuous Batching policy over the packed JetSpec runtime.

The adapter owns only host tickets, queue order, and output-delivery cursors.
Canonical KV, Draft caches, scratch leases, and commit transactions remain
owned by JetSpecBatchRuntime. Arrivals/cancellation occur between step calls;
this is not an asynchronous or thread-safe scheduler.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import count
from time import perf_counter
from typing import Any


@dataclass
class JetSpecTicket:
    request_id: str | int
    prompt_ids: list[int]
    max_new_tokens: int
    tree_budget: int
    ignore_eos: bool
    created_at: float
    arrival_order: int
    status: str = "waiting"
    request: Any = None
    prefill: Any = None
    snapshot: dict | None = None
    output_ids: list[int] = field(default_factory=list)
    emitted_cursor: int = 0
    preemptions: int = 0


class JetSpecScheduler:
    """FIFO admission and round-robin packed decode with recompute preemption.

    Events with kind ``tokens`` contain an incremental token_ids list. Terminal
    events contain the full output, reason and metrics. Cancel returns a
    confirmation of the same terminal event queued for the next step/drain.
    Tickets are forgotten once terminal delivery is queued; IDs may be reused
    after any old pending terminal event has been drained.
    """

    def __init__(self, runtime, max_num_seqs: int, max_verify_tokens: int | None = None,
                 max_admissions_per_step: int = 2, max_prefill_tokens: int | None = None,
                 enable_chunked_prefill: bool = True, prefill_chunk_size: int = 256):
        if max_verify_tokens is None:
            max_verify_tokens = runtime.max_verify_tokens
        if max_prefill_tokens is None:
            max_prefill_tokens = (min(512, runtime.max_verify_tokens)
                                  if enable_chunked_prefill else runtime.max_model_len)
        if not isinstance(enable_chunked_prefill, bool):
            raise ValueError("enable_chunked_prefill must be a boolean")
        for name, value in (("max_num_seqs", max_num_seqs), ("max_verify_tokens", max_verify_tokens),
                            ("max_admissions_per_step", max_admissions_per_step),
                            ("max_prefill_tokens", max_prefill_tokens),
                            ("prefill_chunk_size", prefill_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_verify_tokens > runtime.max_verify_tokens:
            raise ValueError("scheduler verification budget exceeds the runner budget")
        if runtime.requests or getattr(runtime, "prefills", {}):
            raise RuntimeError("a serving scheduler requires an idle borrowed runtime")
        self.runtime = runtime
        self.max_num_seqs = max_num_seqs
        self.max_verify_tokens = max_verify_tokens
        self.max_admissions_per_step = max_admissions_per_step
        self.max_prefill_tokens = max_prefill_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.prefill_chunk_size = prefill_chunk_size
        self.requests: dict[str | int, JetSpecTicket] = {}
        self.waiting: deque[str | int] = deque()
        self.running: deque[str | int] = deque()
        self.prefilling: deque[str | int] = deque()
        self.pending_events: deque[dict] = deque()
        self.last_step: dict | None = None
        self._ids = count()
        self._arrival = count()
        self._in_step = False
        self._closed = False
        self._blocked_reason = None
        self._prefill_chunks: list[dict] = []
        self._prefill_preempted: list[str | int] = []

    def _check_boundary(self):
        if self._closed:
            raise RuntimeError("JetSpec serving scheduler is closed")
        if self._in_step:
            raise RuntimeError("JetSpec arrivals/cancellation require a step boundary")
        self.runtime._check_idle()

    def add_request(self, prompt, *, max_new_tokens: int = 32, tree_budget: int = 63,
                    ignore_eos: bool = False, request_id: str | int | None = None):
        self._check_boundary()
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if isinstance(tree_budget, bool) or not isinstance(tree_budget, int) or not 1 <= tree_budget <= self.runtime.max_tree_budget:
            raise ValueError("tree budget exceeds the configured request cap")
        ids = self.runtime.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not ids or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
            raise ValueError("prompt must contain nonnegative integer token IDs")
        if not self.enable_chunked_prefill and len(ids) > self.max_prefill_tokens:
            raise ValueError("prompt exceeds the nonchunked prefill token budget")
        if max_new_tokens and len(ids) + max_new_tokens + self.runtime.tree_depth > self.runtime.max_model_len:
            raise ValueError("prompt/output/tree lookahead exceed the configured model length")
        if request_id is None:
            request_id = next(self._ids)
            while request_id in self.requests or any(e["request_id"] == request_id for e in self.pending_events):
                request_id = next(self._ids)
        if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
            raise ValueError("request ID must be a string or integer")
        if request_id in self.requests or any(e["request_id"] == request_id for e in self.pending_events):
            raise ValueError("request ID is live or has undelivered events")
        ticket = JetSpecTicket(request_id, ids, max_new_tokens, tree_budget, bool(ignore_eos),
                              perf_counter(), next(self._arrival))
        self.requests[request_id] = ticket
        if max_new_tokens == 0:
            self._terminal(ticket, "finished", "max_tokens_zero")
        else:
            self.waiting.append(request_id)
        return request_id

    def is_finished(self) -> bool:
        # A final cancellation/zero-token completion still needs one step or
        # explicit drain before the engine loop can truthfully stop.
        return not self.requests and not self.pending_events

    def capacity_snapshot(self) -> dict:
        """Physical runner capacity plus host-side admission/recompute queues."""
        return {**self.runtime.capacity_snapshot(), "waiting_count": len(self.waiting),
                "running_count": len(self.running),
                "prefilling_count": len(self.prefilling),
                "suspended_count": sum(ticket.snapshot is not None for ticket in self.requests.values())}

    def drain_events(self) -> list[dict]:
        if self._in_step:
            raise RuntimeError("cannot drain events during a serving step")
        events = list(self.pending_events)
        self.pending_events.clear()
        return events

    @staticmethod
    def _remove(queue, request_id):
        try:
            queue.remove(request_id)
        except ValueError:
            pass

    def _progress(self, ticket):
        if ticket.request is not None:
            ticket.output_ids = list(ticket.request.output_ids)
        elif ticket.snapshot is not None:
            ticket.output_ids = list(ticket.snapshot["output_ids"])
        if len(ticket.output_ids) < ticket.emitted_cursor:
            raise RuntimeError("request output history moved backwards")
        if len(ticket.output_ids) > ticket.emitted_cursor:
            event = {"kind": "tokens", "request_id": ticket.request_id,
                     "token_ids": ticket.output_ids[ticket.emitted_cursor:],
                     "output_length": len(ticket.output_ids)}
            self.pending_events.append(event)
            ticket.emitted_cursor = len(ticket.output_ids)

    def _terminal(self, ticket, kind, reason=None, result=None):
        self._progress(ticket)
        request = ticket.request
        rounds = request.rounds if request is not None else (
            ticket.snapshot.get("rounds", []) if ticket.snapshot is not None else [])
        event = {"kind": kind, "request_id": ticket.request_id,
                 "token_ids": list(ticket.output_ids), "reason": reason,
                 "metrics": {"latency_s": perf_counter() - ticket.created_at,
                             "preemptions": ticket.preemptions, "rounds_count": len(rounds)}}
        if result is not None:
            event["result"] = result
        self.pending_events.append(event)
        self._remove(self.waiting, ticket.request_id)
        self._remove(self.running, ticket.request_id)
        self._remove(self.prefilling, ticket.request_id)
        self.requests.pop(ticket.request_id, None)
        ticket.status = kind
        ticket.request = None
        ticket.prefill = None
        ticket.snapshot = None
        return event

    def _finish_completed(self):
        for request_id in list(self.running):
            ticket = self.requests[request_id]
            if ticket.request.finished:
                try:
                    result = self.runtime.finish(ticket.request)
                    self._terminal(ticket, "finished", "eos_or_output_limit", result)
                except BaseException as exception:
                    try:
                        self._fail(ticket, exception)
                    except BaseException as cleanup_error:
                        exception.add_note(f"finished request cleanup also failed: {cleanup_error}")
                    raise

    def cancel(self, request_id):
        self._check_boundary()
        ticket = self.requests.get(request_id)
        if ticket is None:
            return None
        try:
            if ticket.prefill is not None:
                self.runtime.cancel_prefill(ticket.prefill)
                ticket.prefill = None
            result = self.runtime.cancel(ticket.request) if ticket.request is not None else None
            return self._terminal(ticket, "cancelled", "cancelled_by_caller", result)
        except BaseException as exception:
            try:
                self._fail(ticket, exception)
            except BaseException as cleanup_error:
                exception.add_note(f"cancelled request cleanup also failed: {cleanup_error}")
            raise

    def _fail(self, ticket, exception):
        # Find a runtime request even if create/resume failed after registering
        # it but before returning it to the adapter.
        partial = ticket.prefill or getattr(self.runtime, "prefills", {}).get(ticket.request_id)
        if partial is not None:
            self.runtime.cancel_prefill(partial)
            ticket.prefill = None
        request = ticket.request or self.runtime.requests.get(ticket.request_id)
        result = None
        if request is not None:
            ticket.request = request
            self._progress(ticket)
            result = getattr(request, "result", None)
            if self.runtime.requests.get(ticket.request_id) is request:
                try:
                    result = self.runtime.cancel(request)
                except BaseException as cleanup_error:
                    if self.runtime.requests.get(ticket.request_id) is request:
                        raise
                    # finish() already released canonical pages and published
                    # a result; a later Draft reset error cannot undo release.
                    result = getattr(request, "result", None)
                    exception.add_note(f"released request Draft cleanup also failed: {cleanup_error}")
        return self._terminal(ticket, "error", f"{type(exception).__name__}: {exception}", result)

    def _capacity_error(self, ticket, reason):
        exception = RuntimeError(f"KV capacity exhausted: {reason}")
        return self._fail(ticket, exception)

    def _insert_waiting_by_arrival(self, ticket):
        for position, request_id in enumerate(self.waiting):
            if self.requests[request_id].arrival_order > ticket.arrival_order:
                self.waiting.insert(position, ticket.request_id)
                break
        else:
            self.waiting.append(ticket.request_id)

    def _preempt(self, ticket, preempted_ids):
        self._progress(ticket)
        try:
            snapshot = self.runtime.suspend(ticket.request)
        except BaseException:
            # Runtime prepares/reset Draft before releasing any canonical page.
            # A failed suspension remains a resident, with delivery reconciled.
            self._progress(ticket)
            raise
        ticket.snapshot = snapshot
        ticket.request = None
        ticket.status = "suspended"
        ticket.preemptions += 1
        self._remove(self.running, ticket.request_id)
        self._insert_waiting_by_arrival(ticket)
        preempted_ids.append(ticket.request_id)

    def _clear_excess_scratch(self):
        # The runner is idle between steps. It fences the retired event before
        # freeing pages; request readiness borrows events, never arena ownership.
        if len(self.runtime.arena.blocks) > 1:
            self.runtime.arena.clear()
            return True
        return False

    def _prefill_length(self, ticket):
        return len(ticket.snapshot["committed_tokens"]) - 1 if ticket.snapshot is not None else len(ticket.prompt_ids)

    def _minimum_total_pages(self, cached_tokens, remaining_outputs):
        b = self.runtime.block_size
        # Only current-prefix + one physical committed slot + one scratch page
        # is mandatory. Do not reserve every possible future output in advance.
        return (cached_tokens + (1 if remaining_outputs else 0) + b - 1) // b + bool(remaining_outputs)

    def _preempt_prefill(self, ticket):
        """Discard partial reconstruction, never its delivered output/snapshot.

        Partial requests cannot enter Draft or share tree scratch. Reclaiming a
        younger partial first prevents several incomplete prefixes from holding
        all pages while a resident or the oldest partial cannot make progress.
        """
        self.runtime.cancel_prefill(ticket.prefill)
        ticket.prefill = None
        ticket.preemptions += 1
        ticket.status = "suspended" if ticket.snapshot is not None else "waiting"
        self._remove(self.prefilling, ticket.request_id)
        self._insert_waiting_by_arrival(ticket)
        self._prefill_preempted.append(ticket.request_id)

    def _begin_prefills(self, admitted_ids, preempted_ids):
        attempts = 0
        excluded = set(preempted_ids) | set(self._prefill_preempted)
        while (self.waiting and len(self.running) + len(self.prefilling) < self.max_num_seqs
               and attempts < self.max_admissions_per_step):
            ticket = self.requests[self.waiting[0]]
            # Do not immediately restart a victim of this very step, which
            # would reoccupy the pages just released to allow decode progress.
            if ticket.request_id in excluded:
                break
            # Once a partial/recompute has been evicted, stop admitting younger
            # work until that prefix is READY. Otherwise a sustained stream of
            # short prompts can repeatedly steal its pages: completed younger
            # residents look like progress while the old long prompt starves.
            # Existing residents still decode; their finite completion releases
            # capacity without preempting them just to admit a newcomer.
            retries = [t for t in self.requests.values()
                       if t.preemptions and t.request is None]
            if retries and ticket.arrival_order > min(t.arrival_order for t in retries):
                self._blocked_reason = "prefill_retry_admission_fence"
                break
            length = self._prefill_length(ticket)
            remaining = ticket.max_new_tokens - (len(ticket.output_ids) if ticket.snapshot is not None else 1)
            attempts += 1
            if self._minimum_total_pages(length, remaining) > len(self.runtime.block_manager.blocks):
                self._capacity_error(ticket, "current prefix cannot fit even singleton tree_budget=1")
                continue
            try:
                if ticket.snapshot is not None:
                    partial = self.runtime.begin_prefill(snapshot=ticket.snapshot)
                else:
                    partial = self.runtime.begin_prefill(ticket.prompt_ids,
                        max_new_tokens=ticket.max_new_tokens, tree_budget=ticket.tree_budget,
                        ignore_eos=ticket.ignore_eos, request_id=ticket.request_id)
                    admitted_ids.append(ticket.request_id)
                ticket.prefill = partial
                ticket.status = "prefilling"
                self.waiting.popleft()
                self.prefilling.append(ticket.request_id)
            except BaseException as exception:
                try:
                    self._fail(ticket, exception)
                except BaseException as cleanup_error:
                    exception.add_note(f"partial prefill admission cleanup also failed: {cleanup_error}")
                raise

    def _prefill_can_advance(self, ticket, count):
        while True:
            estimate = self.runtime.estimate_prefill_chunk_capacity(ticket.prefill, count)
            required = estimate["required_free_blocks"]
            free = len(self.runtime.block_manager.free_block_ids)
            if self.running:
                # Reserve enough *free* capacity for one resident's next root
                # verification. No separate scratch is leased for prefilling.
                oldest = self.requests[self.running[0]].request
                keep = self.runtime.estimate_step_capacity([oldest], [1])["required_free_blocks"]
                if required + keep <= free:
                    return True
                self._blocked_reason = "prefill_preserves_resident_decode"
                return False
            if required <= free:
                return True
            if self.runtime.arena.blocks:
                self.runtime.arena.clear()
                continue
            partials = [self.requests[rid] for rid in self.prefilling]
            oldest = min(partials, key=lambda t: t.arrival_order)
            if ticket is oldest:
                victims = [t for t in partials if t is not ticket and t.prefill.owned_blocks]
                if victims:
                    self._preempt_prefill(max(victims, key=lambda t: t.arrival_order))
                    continue
            self._blocked_reason = "waiting_for_kv_prefill"
            return False

    def _advance_prefills(self, admitted_ids, resumed_ids, preempted_ids):
        self._begin_prefills(admitted_ids, preempted_ids)
        remaining = self.max_prefill_tokens
        # One quantum per partial per step; rotate survivors so a small total
        # budget cannot starve the second request behind the first long prompt.
        for request_id in list(self.prefilling):
            if not remaining:
                break
            ticket = self.requests.get(request_id)
            if ticket is None or ticket.prefill is None:
                continue
            partial = ticket.prefill
            start = partial.processed_tokens
            count = min(self.prefill_chunk_size, remaining, partial.total_tokens - start)
            if not self._prefill_can_advance(ticket, count):
                continue
            record = {"request_id": request_id, "start": start, "end": start + count,
                      "total_tokens": partial.total_tokens,
                      "is_recompute": ticket.snapshot is not None, "completed": False}
            try:
                request = self.runtime.prefill_step(partial, count)
                record["completed"] = request is not None
                self._prefill_chunks.append(record)
                remaining -= count
                self._remove(self.prefilling, request_id)
                if request is None:
                    self.prefilling.append(request_id)
                    continue
                ticket.prefill = None
                ticket.request = request
                request.preemptions = ticket.preemptions
                if ticket.snapshot is not None:
                    resumed_ids.append(request_id)
                ticket.snapshot = None
                ticket.status = "running"
                self.running.append(request_id)
                self._progress(ticket)  # Recompute keeps the existing cursor.
                if request.finished:
                    result = self.runtime.finish(request)
                    self._terminal(ticket, "finished", "eos_or_output_limit", result)
            except BaseException as exception:
                try:
                    self._fail(ticket, exception)
                except BaseException as cleanup_error:
                    exception.add_note(f"partial prefill progress cleanup also failed: {cleanup_error}")
                raise

    def _admit(self, admitted_ids, resumed_ids):
        attempts = 0
        prefill_tokens = 0
        while self.waiting and len(self.running) < self.max_num_seqs and attempts < self.max_admissions_per_step:
            ticket = self.requests[self.waiting[0]]
            length = self._prefill_length(ticket)
            remaining = ticket.max_new_tokens - (len(ticket.output_ids) if ticket.snapshot is not None else 1)
            if length > self.max_prefill_tokens:
                self._capacity_error(ticket, "recompute prefix exceeds the nonchunked prefill token budget")
                attempts += 1
                continue
            if prefill_tokens + length > self.max_prefill_tokens:
                break
            if self._minimum_total_pages(length, remaining) > len(self.runtime.block_manager.blocks):
                self._capacity_error(ticket, "current prefix cannot fit even singleton tree_budget=1")
                attempts += 1
                continue
            estimate = self.runtime.estimate_prefill_capacity(length, 1, remaining)
            if not estimate["feasible"] and not self.running:
                # Historical high-water scratch must not block a fresh session.
                if self.runtime.arena.blocks:
                    self.runtime.arena.clear()
                    estimate = self.runtime.estimate_prefill_capacity(length, 1, remaining)
            if not estimate["feasible"]:
                # Never evict a progressing resident solely to admit a newcomer.
                self._blocked_reason = "waiting_for_kv_admission"
                break
            if self.running:
                oldest = self.requests[self.running[0]].request
                keep_decode = self.runtime.estimate_step_capacity([oldest], [1])
                free_after_prefill = len(self.runtime.block_manager.free_block_ids) - estimate["canonical_blocks"]
                if free_after_prefill < keep_decode["required_free_blocks"]:
                    self._blocked_reason = "admission_preserves_resident_decode"
                    break
            attempts += 1
            try:
                if ticket.snapshot is not None:
                    request = self.runtime.resume(ticket.snapshot)
                    resumed_ids.append(ticket.request_id)
                else:
                    request = self.runtime.create_request(ticket.prompt_ids,
                        max_new_tokens=ticket.max_new_tokens, tree_budget=ticket.tree_budget,
                        ignore_eos=ticket.ignore_eos, request_id=ticket.request_id)
                    admitted_ids.append(ticket.request_id)
                ticket.request = request
                ticket.snapshot = None
                ticket.status = "running"
                self.waiting.popleft()
                self.running.append(ticket.request_id)
                prefill_tokens += length
                self._progress(ticket)  # Resume leaves the existing cursor untouched.
                if request.finished:
                    result = self.runtime.finish(request)
                    self._terminal(ticket, "finished", "eos_or_output_limit", result)
            except BaseException as exception:
                try:
                    self._fail(ticket, exception)
                except BaseException as cleanup_error:
                    exception.add_note(f"request cleanup also failed: {cleanup_error}")
                raise

    def _select(self, preempted_ids):
        while self.running:
            tickets = [self.requests[request_id] for request_id in self.running]
            # Every selected request gets one node first. Share the remaining
            # query budget fairly rather than letting the first cap consume it.
            selected = tickets[:min(self.max_num_seqs, self.max_verify_tokens)]
            caps = [1 if t.max_new_tokens - len(t.request.output_ids) == 1 else t.tree_budget
                    for t in selected]
            budgets = [1] * len(selected)
            remaining = self.max_verify_tokens - len(selected)
            eligible = [i for i, cap in enumerate(caps) if cap > 1]
            while remaining and eligible:
                share = max(1, remaining // len(eligible))
                for index in eligible:
                    added = min(caps[index] - budgets[index], share, remaining)
                    budgets[index] += added
                    remaining -= added
                    if not remaining:
                        break
                eligible = [i for i in eligible if budgets[i] < caps[i]]
            requests = [t.request for t in selected]
            if self.runtime.estimate_step_capacity(requests, budgets)["feasible"]:
                return selected, budgets
            # Shrink work before changing ownership. This remains the same
            # packed backend; budget=1 is its root-only serving fallback.
            while any(budget > 1 for budget in budgets):
                index = max(range(len(budgets)), key=budgets.__getitem__)
                budgets[index] = max(1, budgets[index] // 2)
                if self.runtime.estimate_step_capacity(requests, budgets)["feasible"]:
                    return selected, budgets
            while len(selected) > 1:
                selected.pop()
                budgets.pop()
                requests.pop()
                if self.runtime.estimate_step_capacity(requests, budgets)["feasible"]:
                    return selected, budgets
            oldest = selected[0]
            if self._clear_excess_scratch():
                continue
            remaining_outputs = oldest.max_new_tokens - len(oldest.request.output_ids)
            if self._minimum_total_pages(oldest.request.state.cache_len, remaining_outputs) > len(self.runtime.block_manager.blocks):
                self._capacity_error(oldest, "grown prefix cannot fit even singleton tree_budget=1")
                continue
            partials = [self.requests[rid] for rid in self.prefilling
                        if self.requests[rid].prefill.owned_blocks]
            if partials:
                self._preempt_prefill(max(partials, key=lambda t: t.arrival_order))
                continue
            victim = next((t for t in reversed(tickets) if t is not oldest), None)
            if victim is None:
                # A held provisional page can be returned between steps. This
                # is backpressure, not an impossible request or a decoder bug.
                self._blocked_reason = "waiting_for_kv_verification"
                return [], []
            self._preempt(victim, preempted_ids)
        return [], []

    def _report(self, verification, admitted_ids, resumed_ids, preempted_ids, *, events):
        # A page-empty partial registration is not computational progress. If
        # external leases prevent even its first chunk, blocking generate must
        # report backpressure immediately rather than perform a spurious step.
        blocked = bool(self.requests and verification is None and not resumed_ids
                       and not self._prefill_chunks
                       and (self.enable_chunked_prefill or not admitted_ids))
        return {"events": events, "verification": verification,
                "admitted_ids": list(admitted_ids), "resumed_ids": list(resumed_ids),
                "preempted_ids": list(preempted_ids), "waiting_count": len(self.waiting),
                "running_count": len(self.running), "capacity": self.capacity_snapshot(),
                "prefilling_count": len(self.prefilling),
                "prefill_chunks": list(self._prefill_chunks),
                "prefill_tokens": sum(r["end"] - r["start"] for r in self._prefill_chunks),
                "prefill_preempted_ids": list(self._prefill_preempted),
                "blocked": blocked,
                "blocked_reason": (self._blocked_reason or "waiting_for_kv_or_admission") if blocked else None}

    def _decode_once(self, preempted_ids):
        selected, budgets = self._select(preempted_ids)
        if not selected:
            return None
        try:
            verification = self.runtime.step([t.request for t in selected], tree_budgets=budgets)
        except BaseException as exception:
            # Commit may already have published all output/state. Emit that
            # progress exactly once, then terminate only this selected batch.
            for ticket in selected:
                try:
                    self._fail(ticket, exception)
                except BaseException as cleanup_error:
                    exception.add_note(f"request {ticket.request_id} cleanup also failed: {cleanup_error}")
            raise
        for ticket in selected:
            self._progress(ticket)
        self._finish_completed()
        for ticket in selected:
            if ticket.request_id in self.requests:
                self._remove(self.running, ticket.request_id)
                self.running.append(ticket.request_id)
        return verification

    def step(self):
        self._check_boundary()
        self._in_step = True
        admitted_ids, resumed_ids, preempted_ids = [], [], []
        verification = None
        self._blocked_reason = None
        self._prefill_chunks = []
        self._prefill_preempted = []
        try:
            self._finish_completed()
            if self.enable_chunked_prefill:
                # Resident decode has priority. This is interleaved execution,
                # not a fused prefill/tree attention forward or async executor.
                verification = self._decode_once(preempted_ids)
                self._advance_prefills(admitted_ids, resumed_ids, preempted_ids)
                if verification is None:
                    verification = self._decode_once(preempted_ids)
            else:
                # Explicit reference switch retains Phase 4 admission/order.
                self._admit(admitted_ids, resumed_ids)
                verification = self._decode_once(preempted_ids)
            events = list(self.pending_events)
            report = self._report(verification, admitted_ids, resumed_ids, preempted_ids, events=events)
            self.last_step = report
            self.pending_events.clear()
            return report
        except BaseException as exception:
            # Keep undelivered events available to the caller's next step/drain.
            try:
                self.last_step = self._report(verification, admitted_ids, resumed_ids, preempted_ids,
                                              events=list(self.pending_events))
            except BaseException as reporting_error:
                exception.add_note(f"serving error reporting also failed: {reporting_error}")
                self.last_step = {"events": list(self.pending_events), "verification": verification,
                                  "admitted_ids": admitted_ids, "resumed_ids": resumed_ids,
                                  "preempted_ids": preempted_ids, "waiting_count": len(self.waiting),
                                  "running_count": len(self.running), "capacity": None,
                                  "prefilling_count": len(self.prefilling),
                                  "prefill_chunks": list(self._prefill_chunks),
                                  "prefill_tokens": sum(r["end"] - r["start"] for r in self._prefill_chunks),
                                  "prefill_preempted_ids": list(self._prefill_preempted),
                                  "blocked": False, "blocked_reason": None,
                                  "reporting_error": str(reporting_error)}
            raise
        finally:
            self._in_step = False

    def close(self):
        """Cancel tickets and return notifications; never close borrowed runtime."""
        if self._closed:
            return []
        self._check_boundary()
        for request_id in list(self.requests):
            self.cancel(request_id)
        if not self.runtime.requests and not getattr(self.runtime, "prefills", {}):
            self.runtime.release_idle_scratch()
        events = self.drain_events()
        self._closed = True
        return events
