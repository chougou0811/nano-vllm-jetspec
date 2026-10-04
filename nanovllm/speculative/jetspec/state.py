from __future__ import annotations

from dataclasses import dataclass, field

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


def _slots_for_blocks(
    blocks: list[int], length: int, block_size: int, device: torch.device,
    *, start: int = 0,
) -> torch.Tensor:
    """Canonical logical offsets mapped through a layer-shared block table."""
    table = torch.tensor(blocks, dtype=torch.long, device=device)
    positions = torch.arange(start, start + length, dtype=torch.long, device=device)
    return table[positions // block_size] * block_size + positions % block_size


def copy_accepted_kv(
    kv_pool: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    block_size: int,
) -> int:
    """Copy only accepted raw, post-RoPE K/V, across both KV axes and all layers.

    Advanced indexing materializes the selected payload before the scatter, so
    this helper is overlap-safe as well. No prefix gather or RoPE recomputation
    occurs. The returned byte count is payload size, not read + write traffic.
    """
    if source_slots.ndim != 1 or destination_slots.ndim != 1:
        raise ValueError("KV copy requires one-dimensional slot maps")
    if source_slots.numel() != destination_slots.numel():
        raise ValueError("KV copy source and destination lengths differ")
    src_blocks = torch.div(source_slots, block_size, rounding_mode="floor")
    src_offsets = source_slots % block_size
    dst_blocks = torch.div(destination_slots, block_size, rounding_mode="floor")
    dst_offsets = destination_slots % block_size
    accepted = kv_pool[:, :, src_blocks, src_offsets]
    kv_pool[:, :, dst_blocks, dst_offsets] = accepted
    return int(accepted.numel() * accepted.element_size())


@dataclass
class TreeScratchArena:
    """Reusable, private workspace ownership, separate from committed requests.

    Backing pages belong to the arena, never to a prefix cache or accepted path.
    A future packed batch can partition these slots among its requests; c1 uses
    one lease spanning its tree. Reuse waits on the retiring stream's event.
    """

    kv_pool: torch.Tensor
    block_manager: object
    block_size: int
    blocks: list[int] = field(default_factory=list)
    active: bool = False
    _stream: object | None = field(default=None, repr=False)
    _retired: object | None = field(default=None, repr=False)
    _batch_transaction: object | None = field(default=None, repr=False)

    @property
    def capacity(self) -> int:
        return len(self.blocks) * self.block_size

    def wait_ready(self) -> None:
        """Make retired writes visible to the current stream without host wait."""
        if self.kv_pool.is_cuda and self._retired is not None:
            torch.cuda.current_stream(self.kv_pool.device).wait_event(self._retired)

    def acquire(self, n_nodes: int) -> torch.Tensor:
        if self.active:
            raise RuntimeError("tree scratch lease is already active")
        if n_nodes <= 0 or n_nodes > self.capacity:
            raise ValueError("tree does not fit admitted scratch capacity")
        if self.kv_pool.is_cuda:
            stream = torch.cuda.current_stream(self.kv_pool.device)
            self.wait_ready()
            self._stream = stream
        slots = _slots_for_blocks(
            self.blocks, n_nodes, self.block_size, self.kv_pool.device
        )
        self.active = True
        return slots

    def check_stream(self) -> None:
        if self.kv_pool.is_cuda and self.active:
            current = torch.cuda.current_stream(self.kv_pool.device)
            if current != self._stream:
                raise RuntimeError("verify and commit must use the scratch lease stream")

    def retire(self) -> None:
        if not self.active:
            return
        if self.kv_pool.is_cuda:
            # Recording on the lease stream also covers verification exceptions.
            event = torch.cuda.Event()
            event.record(self._stream)
            self._retired = event
        self.active = False
        self._stream = None

    def synchronize(self) -> None:
        if self.kv_pool.is_cuda:
            if self.active:
                self.retire()
            if self._retired is not None:
                self._retired.synchronize()

    def clear(self) -> int:
        if self._batch_transaction is not None:
            raise RuntimeError("abort the active batch transaction before clearing its arena")
        self.retire()
        self.synchronize()
        released = len(self.blocks)
        self.block_manager.release_provisional(self.blocks)
        self.blocks = []
        self._retired = None
        return released


@dataclass
class _TreeRound:
    node_slots: torch.Tensor
    max_path_length: int
    newly_reserved_blocks: int
    destination_blocks: int


@dataclass
class _FeatureAppendPlan:
    hidden: torch.Tensor
    storage: torch.Tensor | None
    append_bytes: int
    history_bytes: int
    next_append_bytes: int
    next_history_bytes: int
    next_growths: int


@dataclass
class PagedTargetState:
    """Canonical committed KV plus one bounded, reusable tree transaction arena."""

    committed: torch.Tensor
    target_hidden: torch.Tensor
    kv_pool: torch.Tensor
    block_manager: object
    block_size: int
    logical_slots: torch.Tensor
    owned_blocks: list[int]
    pending_blocks: list[int]
    scratch: TreeScratchArena | None = None
    _round: _TreeRound | None = field(default=None, repr=False)
    _batch_transaction: object | None = field(default=None, repr=False)
    _cleared: bool = field(default=False, repr=False)
    validate_device: bool = True
    _feature_storage: torch.Tensor | None = field(default=None, repr=False)
    _feature_max_capacity: int | None = field(default=None, repr=False)
    _feature_append_bytes: int = field(default=0, repr=False)
    _feature_history_bytes: int = field(default=0, repr=False)
    _feature_growths: int = field(default=0, repr=False)

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
        if block_size <= 0 or kv_pool.ndim != 6 or kv_pool.shape[0] != 2:
            raise ValueError("invalid KV pool geometry")
        if kv_pool.shape[3] != block_size or block_manager.block_size != block_size:
            raise ValueError("KV pool, allocator and state page geometries differ")
        if len(prompt_key_values) != kv_pool.shape[1]:
            raise ValueError("prefill KV layer count does not match the pool")
        cache_len = int(prompt_key_values[0][0].shape[0])
        if committed.ndim != 2 or committed.shape[0] != 1:
            raise ValueError("c1 committed tokens must have shape [1, tokens]")
        if committed.shape[1] - 1 != cache_len or target_hidden.shape[1] != cache_len:
            raise ValueError("prefill KV, feature and committed token lengths differ")
        expected_shape = (cache_len, int(kv_pool.shape[4]), int(kv_pool.shape[5]))
        for keys, values in prompt_key_values:
            if tuple(keys.shape) != expected_shape or tuple(values.shape) != expected_shape:
                raise ValueError("prefill KV shape does not match the pool")
        blocks = block_manager.reserve_provisional((cache_len + block_size - 1) // block_size)
        try:
            slots = _slots_for_blocks(blocks, cache_len, block_size, kv_pool.device)
            physical_blocks = torch.div(slots, block_size, rounding_mode="floor")
            offsets = slots % block_size
            for layer_id, (keys, values) in enumerate(prompt_key_values):
                kv_pool[0, layer_id, physical_blocks, offsets] = keys
                kv_pool[1, layer_id, physical_blocks, offsets] = values
            scratch = TreeScratchArena(kv_pool, block_manager, int(block_size))
            if kv_pool.is_cuda:
                # The first round or clear may run on another stream. Retaining
                # prefill's completion dependency protects its committed pages.
                scratch._retired = torch.cuda.Event()
                scratch._retired.record(torch.cuda.current_stream(kv_pool.device))
            return cls(
                committed=committed,
                target_hidden=target_hidden,
                kv_pool=kv_pool,
                block_manager=block_manager,
                block_size=int(block_size),
                logical_slots=slots,
                owned_blocks=blocks,
                pending_blocks=[],
                scratch=scratch,
            )
        except BaseException:
            # Scatter may already have queued writes into pages being returned.
            if kv_pool.is_cuda:
                torch.cuda.current_stream(kv_pool.device).synchronize()
            block_manager.release_provisional(blocks)
            raise

    @property
    def cache_len(self) -> int:
        return int(self.logical_slots.numel())

    @property
    def feature_capacity(self) -> int:
        buffer = self._feature_storage if self._feature_storage is not None else self.target_hidden
        return int(buffer.shape[1]) if buffer.ndim == 3 else 0

    def feature_storage_snapshot(self) -> dict[str, int | bool]:
        buffer = self._feature_storage if self._feature_storage is not None else self.target_hidden
        return {"feature_storage_enabled": self._feature_storage is not None,
                "feature_capacity_tokens": self.feature_capacity,
                "feature_live_tokens": int(self.target_hidden.shape[1]) if self.target_hidden.ndim == 3 else 0,
                "feature_reserved_bytes": int(buffer.numel() * buffer.element_size()),
                "feature_append_copy_bytes": self._feature_append_bytes,
                "feature_history_copy_bytes": self._feature_history_bytes,
                "feature_copy_bytes": self._feature_append_bytes + self._feature_history_bytes,
                "feature_growths": self._feature_growths}

    def enable_feature_storage(self, initial_capacity: int | None = None,
                               max_capacity: int | None = None) -> dict:
        """Opt into append-only feature capacity without changing the tensor API.

        The default reuses the current feature tensor as the initial backing;
        geometric growth copies history only when capacity is exhausted. An
        explicit capacity can reserve a request's known output limit up front.
        Commits write only the not-yet-visible tail, then publish a longer view.
        Old visible views/prefix bytes therefore survive preparation or abort.
        Legacy states retain the original torch.cat behavior by default.
        """
        if self._cleared:
            raise RuntimeError("cannot enable feature storage for a cleared request")
        self.assert_round_invariant()
        if self.target_hidden.ndim != 3 or self.target_hidden.shape[0] != 1:
            raise ValueError("target features must have shape [1, tokens, width]")
        if self._feature_storage is not None:
            if initial_capacity is not None or max_capacity is not None:
                raise RuntimeError("feature storage is already enabled")
            return self.feature_storage_snapshot()
        length = int(self.target_hidden.shape[1])
        capacity = length if initial_capacity is None else initial_capacity
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < length:
            raise ValueError("initial feature capacity cannot be smaller than live history")
        if max_capacity is not None and (isinstance(max_capacity, bool) or not isinstance(max_capacity, int)
                                          or max_capacity < capacity):
            raise ValueError("maximum feature capacity cannot be smaller than initial capacity")
        buffer = self.target_hidden
        copied = 0
        if capacity > length:
            buffer = torch.empty((1, capacity, self.target_hidden.shape[2]),
                                 dtype=self.target_hidden.dtype, device=self.target_hidden.device)
            buffer[:, :length].copy_(self.target_hidden)
            copied = int(self.target_hidden.numel() * self.target_hidden.element_size())
            if self.kv_pool.is_cuda and self.scratch is not None:
                # assert_round_invariant waited for the old readiness event.
                # Retain this feature copy's new dependency for a different
                # stream's first Draft/read or request cleanup.
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self.kv_pool.device))
                self.scratch._retired = event
        self._feature_storage = buffer
        self._feature_max_capacity = max_capacity
        self._feature_history_bytes += copied
        self._feature_growths += int(capacity > length)
        self.target_hidden = buffer[:, :length]
        return self.feature_storage_snapshot()

    def _prepare_feature_append(self, payload: torch.Tensor) -> _FeatureAppendPlan:
        if (payload.ndim != 3 or tuple(payload.shape[:1]) != (1,)
                or payload.shape[2:] != self.target_hidden.shape[2:]
                or (self._feature_storage is not None and payload.dtype != self.target_hidden.dtype)
                or payload.device != self.target_hidden.device):
            raise ValueError("accepted features and committed feature geometry differ")
        length = int(self.target_hidden.shape[1])
        next_length = length + int(payload.shape[1])
        append_bytes = int(payload.numel() * payload.element_size())
        history_bytes = 0
        growths = self._feature_growths
        buffer = self._feature_storage
        if buffer is None:
            hidden = torch.cat((self.target_hidden, payload), dim=1)
            history_bytes = int(self.target_hidden.numel() * self.target_hidden.element_size())
        else:
            if next_length > buffer.shape[1]:
                capacity = max(next_length, max(1, int(buffer.shape[1]) * 2))
                if self._feature_max_capacity is not None:
                    if next_length > self._feature_max_capacity:
                        raise RuntimeError("accepted features exceed the configured capacity limit")
                    capacity = min(capacity, self._feature_max_capacity)
                buffer = torch.empty((1, capacity, self.target_hidden.shape[2]),
                                     dtype=self.target_hidden.dtype, device=self.target_hidden.device)
                buffer[:, :length].copy_(self.target_hidden)
                history_bytes = int(self.target_hidden.numel() * self.target_hidden.element_size())
                growths += 1
            # This region is not part of the old published target_hidden view.
            # A failed transaction can leave arbitrary bytes here; the next
            # preparation overwrites them before making them reachable.
            buffer[:, length:next_length].copy_(payload)
            hidden = buffer[:, :next_length]
        return _FeatureAppendPlan(hidden, buffer, append_bytes, history_bytes,
                                  self._feature_append_bytes + append_bytes,
                                  self._feature_history_bytes + history_bytes, growths)

    def _publish_features(self, plan: _FeatureAppendPlan) -> None:
        self.target_hidden = plan.hidden
        self._feature_storage = plan.storage
        self._feature_append_bytes = plan.next_append_bytes
        self._feature_history_bytes = plan.next_history_bytes
        self._feature_growths = plan.next_growths

    @property
    def scratch_blocks(self) -> list[int]:
        return list(self.scratch.blocks) if self.scratch is not None else []

    @property
    def scratch_active(self) -> bool:
        return self.scratch is not None and self.scratch.active

    def capacity_snapshot(self) -> dict[str, int | float]:
        scratch_count = len(self.scratch_blocks)
        total = len(self.owned_blocks) + len(self.pending_blocks) + scratch_count
        reserved_slots = total * self.block_size
        live = self.cache_len
        return {
            "committed_blocks": len(self.owned_blocks),
            "scratch_blocks": scratch_count,
            "pending_destination_blocks": len(self.pending_blocks),
            "reserved_destination_blocks": len(self.pending_blocks),
            "used_blocks": len(self.block_manager.used_block_ids),
            "reserved_slots": reserved_slots,
            "reserved_kv_slots": reserved_slots,
            "live_slots": live,
            "live_kv_slots": live,
            "amplification": reserved_slots / live if live else 0.0,
        }

    def reserve_tree(
        self, n_nodes: int, max_path_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Atomically admit both worst-case destination growth and scratch.

        The runtime supplies depth + 1 (root-inclusive). The default is the
        conservative bound n_nodes, useful for generic allocator replays.
        """
        if self._cleared:
            raise RuntimeError("cannot reserve a tree for a cleared request")
        if self._batch_transaction is not None:
            raise RuntimeError("request belongs to an active batch transaction")
        if self._round is not None or self.pending_blocks or self.scratch_active:
            raise RuntimeError("previous provisional tree transaction is still active")
        n_nodes = int(n_nodes)
        max_path_length = n_nodes if max_path_length is None else int(max_path_length)
        if n_nodes <= 0 or not 1 <= max_path_length <= n_nodes:
            raise ValueError("invalid tree size or maximum accepted path length")
        if self.scratch is None:
            self.scratch = TreeScratchArena(self.kv_pool, self.block_manager, self.block_size)
        required_destination = (self.cache_len + max_path_length + self.block_size - 1) // self.block_size
        destination_count = required_destination - len(self.owned_blocks)
        scratch_count = (n_nodes + self.block_size - 1) // self.block_size
        scratch_growth = max(0, scratch_count - len(self.scratch.blocks))
        # One allocator call means neither lease can be admitted without the other.
        new_blocks = self.block_manager.reserve_provisional(destination_count + scratch_growth)
        old_scratch_count = len(self.scratch.blocks)
        try:
            destination = new_blocks[:destination_count]
            self.scratch.blocks.extend(new_blocks[destination_count:])
            node_slots = self.scratch.acquire(n_nodes)
            logical = torch.cat((self.logical_slots, node_slots))
            self.pending_blocks = destination
            self._round = _TreeRound(
                node_slots, max_path_length, len(new_blocks), destination_count,
            )
            return node_slots, logical
        except BaseException:
            self.scratch.retire()
            self.scratch.synchronize()
            self.scratch.blocks = self.scratch.blocks[:old_scratch_count]
            self.block_manager.release_provisional(new_blocks)
            self.pending_blocks = []
            self._round = None
            raise

    def commit_tree_path(
        self,
        node_slots: torch.Tensor,
        node_hidden: torch.Tensor,
        accepted_path: torch.Tensor,
        *,
        committed_tokens: torch.Tensor | None = None,
    ) -> dict[str, int | float]:
        """Copy accepted-only KV to the canonical tail, then publish metadata.

        If preparation or copy raises, committed metadata remains unchanged.
        abort_tree()/clear() fence queued writes and reclaim the reservation.
        committed_tokens is the complete new history including correction;
        correction has no destination slot. Omitting it preserves the old API,
        whose caller must publish tokens before asserting the round invariant.
        """
        transaction = self._round
        if self._batch_transaction is not None:
            raise RuntimeError("shared tree scratch must be committed by its batch transaction")
        if transaction is None or self.scratch is None or not self.scratch.active:
            raise RuntimeError("no provisional tree transaction to commit")
        self.scratch.check_stream()
        if node_slots.shape != transaction.node_slots.shape or not torch.equal(node_slots, transaction.node_slots):
            raise ValueError("node slots do not belong to the active tree transaction")
        if accepted_path.ndim != 1:
            raise ValueError("accepted path must be one-dimensional")
        selected = accepted_path.to(device=node_slots.device, dtype=torch.long)
        path = selected.tolist()
        if not path or path[0] != 0 or len(set(path)) != len(path):
            raise ValueError("accepted path must be root-inclusive and contain unique nodes")
        if min(path) < 0 or max(path) >= node_slots.numel():
            raise ValueError("accepted path contains an out-of-range tree node")
        if len(path) > transaction.max_path_length:
            raise ValueError("accepted path exceeds admitted destination capacity")
        if node_hidden.ndim != 3 or node_hidden.shape[0] != 1 or node_hidden.shape[1] != node_slots.numel():
            raise ValueError("tree features do not match the active tree")
        next_length = self.cache_len + len(path)
        if committed_tokens is not None:
            if committed_tokens.ndim != 2 or tuple(committed_tokens.shape) != (1, next_length + 1):
                raise ValueError("published tokens must include exactly one uncached correction")
        combined_blocks = self.owned_blocks + self.pending_blocks
        source = node_slots.index_select(0, selected)
        destination = _slots_for_blocks(
            combined_blocks, len(path), self.block_size, self.kv_pool.device,
            start=self.cache_len,
        )
        next_slots = torch.cat((self.logical_slots, destination))
        feature_plan = self._prepare_feature_append(node_hidden.index_select(1, selected))
        needed_blocks = (next_length + self.block_size - 1) // self.block_size
        used_destination = needed_blocks - len(self.owned_blocks)
        retained = self.pending_blocks[:used_destination]
        unused = self.pending_blocks[used_destination:]
        next_owned_blocks = self.owned_blocks + retained
        copy_bytes = copy_accepted_kv(self.kv_pool, source, destination, self.block_size)
        # No unused destination has been written. All potentially live writes are
        # covered by the retiring event before scratch can be acquired again.
        self.scratch.retire()
        self.block_manager.release_provisional(unused)
        self.owned_blocks = next_owned_blocks
        self.pending_blocks = []
        self.logical_slots = next_slots
        self._publish_features(feature_plan)
        if committed_tokens is not None:
            self.committed = committed_tokens
        self._round = None
        return {
            **self.capacity_snapshot(),
            "reserved_blocks": transaction.newly_reserved_blocks,
            "released_blocks": len(unused),
            "reserved_destination_blocks": transaction.destination_blocks,
            "rejected_logical_slots": int(node_slots.numel() - len(path)),
            "accepted_kv_slots": len(path),
            "kv_copy_bytes": copy_bytes,
            **self.feature_storage_snapshot(),
            "feature_append_copy_bytes": feature_plan.append_bytes,
            "feature_history_copy_bytes": feature_plan.history_bytes,
            "feature_copy_bytes": feature_plan.append_bytes + feature_plan.history_bytes,
        }

    def abort_tree(self) -> int:
        """Discard the round without publishing any committed metadata."""
        if self._batch_transaction is not None:
            raise RuntimeError("shared tree scratch must be aborted by its batch transaction")
        if self._round is None:
            return 0
        if self.scratch is not None:
            self.scratch.retire()
            # A failed copy may have written pending destination pages. They must
            # not reenter the global allocator until those writes are complete.
            if self.pending_blocks:
                self.scratch.synchronize()
        released = len(self.pending_blocks)
        self.block_manager.release_provisional(self.pending_blocks)
        self.pending_blocks = []
        self._round = None
        return released

    def assert_round_invariant(self, *, validate_device: bool | None = None) -> None:
        if self.scratch is not None:
            self.scratch.wait_ready()
        expected = int(self.committed.shape[1]) - 1
        if self.cache_len != expected or int(self.target_hidden.shape[1]) != expected:
            raise RuntimeError(
                "JetSpec paged state invariant failed: "
                f"committed-1={expected}, logical_slots={self.cache_len}, "
                f"target_hidden={self.target_hidden.shape[1]}"
            )
        if self._round is not None or self._batch_transaction is not None or self.pending_blocks or self.scratch_active:
            raise RuntimeError("provisional tree transaction survived a completed round")
        expected_blocks = (self.cache_len + self.block_size - 1) // self.block_size
        if len(self.owned_blocks) != expected_blocks:
            raise RuntimeError("committed pages are not tightly allocated")
        check_device = self.validate_device if validate_device is None else validate_device
        if check_device:
            canonical = _slots_for_blocks(
                self.owned_blocks, self.cache_len, self.block_size, self.kv_pool.device,
            )
            if not torch.equal(canonical, self.logical_slots):
                raise RuntimeError("committed slot map is not canonical")
        if self._feature_storage is not None:
            if (self._feature_storage.ndim != 3 or self.target_hidden.ndim != 3
                    or self._feature_storage.shape[0] != 1
                    or self._feature_storage.shape[1] < expected
                    or self._feature_storage.shape[2:] != self.target_hidden.shape[2:]
                    or self._feature_storage.dtype != self.target_hidden.dtype
                    or self._feature_storage.device != self.target_hidden.device
                    or self.target_hidden.untyped_storage().data_ptr() != self._feature_storage.untyped_storage().data_ptr()
                    or self.target_hidden.storage_offset() != self._feature_storage.storage_offset()
                    or self.target_hidden.stride() != self._feature_storage.stride()):
                raise RuntimeError("committed feature view differs from its backing capacity")
        all_blocks = self.owned_blocks + self.scratch_blocks
        if len(set(all_blocks)) != len(all_blocks):
            raise RuntimeError("committed and tree scratch block ownership overlaps")
        for block_id in all_blocks:
            block = self.block_manager.blocks[block_id]
            if block.ref_count != 1 or block_id not in self.block_manager.used_block_ids:
                raise RuntimeError(f"lost paged KV ownership for block {block_id}")
            if block.hash != -1 or block.token_ids:
                raise RuntimeError(f"private JetSpec page entered the prefix cache: {block_id}")

    def clear(self) -> int:
        if self._cleared:
            return 0
        if self._batch_transaction is not None:
            raise RuntimeError("abort the active batch transaction before clearing its request")
        aborted_released = self.abort_tree()
        # This also fences prefill-only requests, which have not retired scratch.
        if self.kv_pool.is_cuda:
            torch.cuda.current_stream(self.kv_pool.device).synchronize()
        scratch_released = self.scratch.clear() if self.scratch is not None else 0
        released = len(self.owned_blocks) + scratch_released + aborted_released
        self.block_manager.release_provisional(self.owned_blocks)
        self.pending_blocks = []
        self.owned_blocks = []
        self.logical_slots = torch.empty(0, dtype=torch.long, device=self.kv_pool.device)
        self.target_hidden = torch.empty(0, device=self.kv_pool.device)
        self._feature_storage = None
        self._feature_max_capacity = None
        self._feature_append_bytes = 0
        self._feature_history_bytes = 0
        self._feature_growths = 0
        self.committed = torch.empty(0, dtype=torch.long, device=self.kv_pool.device)
        self._cleared = True
        return released


@dataclass
class _BatchCommitPlan:
    source: torch.Tensor
    destination: torch.Tensor
    next_slots: torch.Tensor
    next_hidden: torch.Tensor
    next_tokens: torch.Tensor
    next_owned_blocks: list[int]
    unused_blocks: list[int]
    accepted_count: int
    copy_bytes: int
    features: _FeatureAppendPlan


@dataclass
class BatchTreeTransaction:
    """One atomic packed-tree round borrowing a runner-owned scratch arena.

    Requests own only canonical committed pages and pending destination pages.
    They never own, retire, or free the shared arena's pages. All destination
    capacity and arena growth are admitted together before verification. One
    accepted-only all-layer copy precedes publication of every request's state.
    """

    states: list[PagedTargetState]
    arena: TreeScratchArena
    node_slots: list[torch.Tensor]
    packed_node_slots: torch.Tensor
    ranges: list[tuple[int, int]]
    max_path_lengths: list[int]
    destination_counts: list[int]
    newly_reserved_blocks: int
    active: bool = True
    committed: bool = False
    _published: bool = field(default=False, repr=False)
    result: dict | None = field(default=None, repr=False)

    @property
    def logical_slots(self) -> list[torch.Tensor]:
        """Optional diagnostic maps; production packed attention uses page tables."""
        return [torch.cat((state.logical_slots, nodes)) for state, nodes in zip(self.states, self.node_slots)]

    @classmethod
    def admit(
        cls,
        states: list[PagedTargetState],
        node_counts: list[int],
        max_path_lengths: list[int],
        arena: TreeScratchArena,
    ) -> "BatchTreeTransaction":
        states = list(states)
        counts = [int(count) for count in node_counts]
        maxima = [int(length) for length in max_path_lengths]
        if not states or len(states) != len(counts) or len(states) != len(maxima):
            raise ValueError("batch requests, node counts and path bounds must align")
        if len({id(state) for state in states}) != len(states):
            raise ValueError("a request cannot occur twice in one packed batch")
        if arena.active or arena._batch_transaction is not None:
            raise RuntimeError("runner tree scratch is already leased")
        all_owned = []
        destination_counts = []
        for state, n_nodes, maximum in zip(states, counts, maxima):
            if state._cleared:
                raise RuntimeError("cannot admit a cleared request")
            if state.kv_pool is not arena.kv_pool or state.block_manager is not arena.block_manager or state.block_size != arena.block_size:
                raise ValueError("batch requests and arena must use one KV pool and allocator")
            if n_nodes <= 0 or not 1 <= maximum <= n_nodes:
                raise ValueError("invalid packed tree size or maximum accepted path length")
            state.assert_round_invariant()
            all_owned.extend(state.owned_blocks + state.scratch_blocks)
            needed = (state.cache_len + maximum + arena.block_size - 1) // arena.block_size
            destination_counts.append(needed - len(state.owned_blocks))
        if len(set(all_owned)) != len(all_owned) or set(all_owned).intersection(arena.blocks):
            raise RuntimeError("batch request and arena ownership overlap")
        total_nodes = sum(counts)
        old_arena_count = len(arena.blocks)
        scratch_growth = max(0, (total_nodes + arena.block_size - 1) // arena.block_size - old_arena_count)
        total_destinations = sum(destination_counts)
        new_blocks = arena.block_manager.reserve_provisional(total_destinations + scratch_growth)
        try:
            arena.blocks.extend(new_blocks[total_destinations:])
            packed_nodes = arena.acquire(total_nodes)
            ranges = []
            nodes = []
            offset = 0
            for count in counts:
                ranges.append((offset, offset + count))
                nodes.append(packed_nodes[offset:offset + count])
                offset += count
            transaction = cls(
                states, arena, nodes, packed_nodes, ranges, maxima,
                destination_counts, len(new_blocks),
            )
            offset = 0
            for state, count in zip(states, destination_counts):
                state.pending_blocks = new_blocks[offset:offset + count]
                state._batch_transaction = transaction
                offset += count
            arena._batch_transaction = transaction
            return transaction
        except BaseException:
            arena.retire()
            arena.synchronize()
            arena.blocks = arena.blocks[:old_arena_count]
            for state in states:
                state.pending_blocks = []
                state._batch_transaction = None
            arena._batch_transaction = None
            arena.block_manager.release_provisional(new_blocks)
            raise

    def capacity_snapshot(self) -> dict[str, int | float]:
        committed = sum(len(state.owned_blocks) for state in self.states)
        pending = sum(len(state.pending_blocks) for state in self.states)
        private_scratch = sum(len(state.scratch_blocks) for state in self.states)
        reserved = (committed + pending + private_scratch + len(self.arena.blocks)) * self.arena.block_size
        live = sum(state.cache_len for state in self.states)
        return {
            "committed_blocks": committed,
            "scratch_blocks": len(self.arena.blocks),
            "private_scratch_blocks": private_scratch,
            "pending_destination_blocks": pending,
            "reserved_destination_blocks": pending,
            "used_blocks": len(self.arena.block_manager.used_block_ids),
            "reserved_slots": reserved,
            "reserved_kv_slots": reserved,
            "live_slots": live,
            "live_kv_slots": live,
            "amplification": reserved / live if live else 0.0,
        }

    def _prepare_commit(
        self,
        state: PagedTargetState,
        nodes: torch.Tensor,
        node_hidden: torch.Tensor,
        path: torch.Tensor,
        maximum: int,
        tokens: torch.Tensor,
        *,
        accepted_path_host: list[int] | tuple[int, ...] | None = None,
    ) -> _BatchCommitPlan:
        if path.ndim != 1:
            raise ValueError("accepted path must be one-dimensional")
        if accepted_path_host is None:
            selected = path.to(device=nodes.device, dtype=torch.long)
            indices = selected.tolist()
        else:
            # The host path is authoritative, not a hint about another GPU
            # tensor's values. Reconstruct source indices from the validated
            # list so a stale/mismatched device tensor cannot address wrong KV.
            indices = list(accepted_path_host)
            if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
                raise ValueError("accepted host path must contain integer node indices")
            if path.numel() != len(indices):
                raise ValueError("accepted host and supplied path lengths differ")
            selected = torch.tensor(indices, dtype=torch.long, device=nodes.device)
        if not indices or indices[0] != 0 or len(set(indices)) != len(indices):
            raise ValueError("accepted path must be root-inclusive and contain unique nodes")
        if min(indices) < 0 or max(indices) >= nodes.numel() or len(indices) > maximum:
            raise ValueError("accepted path exceeds admitted tree or destination bounds")
        if node_hidden.ndim != 3 or tuple(node_hidden.shape[:2]) != (1, nodes.numel()):
            raise ValueError("tree features do not match their request's tree")
        next_length = state.cache_len + len(indices)
        if tokens.ndim != 2 or tuple(tokens.shape) != (1, next_length + 1):
            raise ValueError("published tokens must include exactly one uncached correction")
        source = nodes.index_select(0, selected)
        destination = _slots_for_blocks(
            state.owned_blocks + state.pending_blocks, len(indices),
            state.block_size, state.kv_pool.device, start=state.cache_len,
        )
        next_slots = torch.cat((state.logical_slots, destination))
        feature_plan = state._prepare_feature_append(node_hidden.index_select(1, selected))
        needed = (next_length + state.block_size - 1) // state.block_size
        used_destination = needed - len(state.owned_blocks)
        retained = state.pending_blocks[:used_destination]
        unused = state.pending_blocks[used_destination:]
        # Every allocation/preparation that can fail happens before the copy.
        next_owned = state.owned_blocks + retained
        bytes_per_slot = int(state.kv_pool.shape[0] * state.kv_pool.shape[1] * state.kv_pool.shape[4] * state.kv_pool.shape[5] * state.kv_pool.element_size())
        return _BatchCommitPlan(
            source, destination, next_slots, feature_plan.hidden, tokens, next_owned,
            unused, len(indices), len(indices) * bytes_per_slot, feature_plan,
        )

    def commit(
        self,
        node_hidden: list[torch.Tensor],
        accepted_paths: list[torch.Tensor],
        committed_tokens: list[torch.Tensor],
        *,
        accepted_paths_host: list[list[int] | tuple[int, ...]] | None = None,
    ) -> dict:
        """Commit validated paths; optional host lists are authoritative indices.

        The default retains full device-to-host path validation. An internal
        batched acceptance extractor may supply already-materialized CPU lists;
        these still receive all host root/uniqueness/bounds/shape checks, and are
        used to construct the actual index tensors (never trust unchecked GPU
        values that merely claim to match them).
        """
        if not self.active or self.committed:
            raise RuntimeError("packed tree transaction is no longer active")
        self.arena.check_stream()
        if len(node_hidden) != len(self.states) or len(accepted_paths) != len(self.states) or len(committed_tokens) != len(self.states):
            raise ValueError("batch commit inputs must align with admitted requests")
        if accepted_paths_host is not None and len(accepted_paths_host) != len(self.states):
            raise ValueError("accepted host paths must align with admitted requests")
        host_paths = [None] * len(self.states) if accepted_paths_host is None else accepted_paths_host
        plans = [
            self._prepare_commit(state, nodes, hidden, path, maximum, tokens,
                                 accepted_path_host=host)
            for state, nodes, hidden, path, maximum, tokens, host in zip(
                self.states, self.node_slots, node_hidden, accepted_paths,
                self.max_path_lengths, committed_tokens, host_paths,
            )
        ]
        source = torch.cat([plan.source for plan in plans])
        destination = torch.cat([plan.destination for plan in plans])
        unused = [block for plan in plans for block in plan.unused_blocks]
        request_metrics = [
            {
                "accepted_kv_slots": plan.accepted_count,
                "kv_copy_bytes": plan.copy_bytes,
                "rejected_logical_slots": int(nodes.numel()) - plan.accepted_count,
                "reserved_destination_blocks": count,
                "released_blocks": len(plan.unused_blocks),
                "committed_blocks": len(plan.next_owned_blocks),
                "live_kv_slots": int(plan.next_slots.numel()),
                "feature_storage_enabled": plan.features.storage is not None,
                "feature_capacity_tokens": int((plan.features.storage if plan.features.storage is not None else plan.next_hidden).shape[1]),
                "feature_reserved_bytes": int((plan.features.storage if plan.features.storage is not None else plan.next_hidden).numel() * plan.next_hidden.element_size()),
                "feature_append_copy_bytes": plan.features.append_bytes,
                "feature_history_copy_bytes": plan.features.history_bytes,
                "feature_copy_bytes": plan.features.append_bytes + plan.features.history_bytes,
                "feature_growths": plan.features.next_growths,
            }
            for plan, nodes, count in zip(plans, self.node_slots, self.destination_counts)
        ]
        original = [
            (state.owned_blocks, state.pending_blocks, state.logical_slots,
             state.target_hidden, state.committed, state._feature_storage,
             state._feature_append_bytes, state._feature_history_bytes, state._feature_growths)
            for state in self.states
        ]
        committed_blocks = sum(len(plan.next_owned_blocks) for plan in plans)
        private_scratch = sum(len(state.scratch_blocks) for state in self.states)
        reserved_slots = (committed_blocks + private_scratch + len(self.arena.blocks)) * self.arena.block_size
        live_slots = sum(int(plan.next_slots.numel()) for plan in plans)
        result = {
            "committed_blocks": committed_blocks,
            "scratch_blocks": len(self.arena.blocks),
            "private_scratch_blocks": private_scratch,
            "pending_destination_blocks": 0,
            "used_blocks": len(self.arena.block_manager.used_block_ids) - len(unused),
            "reserved_slots": reserved_slots,
            "reserved_kv_slots": reserved_slots,
            "live_slots": live_slots,
            "live_kv_slots": live_slots,
            "amplification": reserved_slots / live_slots if live_slots else 0.0,
            "request_metrics": request_metrics,
            "kv_copy_bytes": sum(plan.copy_bytes for plan in plans),
            "accepted_kv_slots": sum(plan.accepted_count for plan in plans),
            "rejected_logical_slots": sum(int(nodes.numel()) - plan.accepted_count for plan, nodes in zip(plans, self.node_slots)),
            "reserved_blocks": self.newly_reserved_blocks,
            "reserved_destination_blocks": sum(self.destination_counts),
            "released_blocks": len(unused),
            "feature_append_copy_bytes": sum(plan.features.append_bytes for plan in plans),
            "feature_history_copy_bytes": sum(plan.features.history_bytes for plan in plans),
            "feature_copy_bytes": sum(plan.features.append_bytes + plan.features.history_bytes for plan in plans),
            "feature_reserved_bytes": sum(metric["feature_reserved_bytes"] for metric in request_metrics),
        }
        self.result = result
        copy_bytes = copy_accepted_kv(self.arena.kv_pool, source, destination, self.arena.block_size)
        if copy_bytes != result["kv_copy_bytes"]:
            raise RuntimeError("accepted KV copy byte count differs from prepared batch payload")
        self.arena.retire()
        # Publish without freeing anything first, so an interrupted publication
        # can restore all old metadata and let abort() reclaim every destination.
        try:
            for state, plan in zip(self.states, plans):
                state.owned_blocks = plan.next_owned_blocks
                state.pending_blocks = []
                state.logical_slots = plan.next_slots
                state._publish_features(plan.features)
                state.committed = plan.next_tokens
            self._published = True
            self.arena.block_manager.release_provisional(unused)
            self.committed = True
        except BaseException:
            # An interruption can land after release_provisional completed but
            # before this frame received its return. The allocator guarantees
            # either a full rollback (all these pages still used) or a completed
            # release. Once all metadata is published and all unused pages have
            # been released, the round is terminal: never restore old ownership.
            released = self._published and all(
                block_id not in self.arena.block_manager.used_block_ids
                and self.arena.block_manager.blocks[block_id].ref_count == 0
                for block_id in unused
            )
            if released:
                self.committed = True
                self.finish_committed()
                raise
            for state, previous in zip(self.states, original):
                (state.owned_blocks, state.pending_blocks, state.logical_slots,
                 state.target_hidden, state.committed, state._feature_storage,
                 state._feature_append_bytes, state._feature_history_bytes, state._feature_growths) = previous
            self._published = False
            raise
        self.finish_committed()
        return result

    def finish_committed(self) -> None:
        """Idempotently finish guards after the irreversible publication point.

        ``committed`` stays true even if one interruption lands during cleanup.
        The caller can publish its precomputed outputs and retry this method (or
        abort()). This is not a guarantee against repeated asynchronous signals
        interrupting the recovery itself or a process being killed outright.
        """
        if not self.committed:
            raise RuntimeError("cannot finalize an uncommitted packed transaction")
        if not self.active:
            return
        for state in self.states:
            if state._batch_transaction is self:
                # Publish readiness before allowing independent request clear.
                # Borrow only the event, never arena page ownership.
                if state.scratch is not None:
                    state.scratch._retired = self.arena._retired
                state._batch_transaction = None
        if self.arena._batch_transaction is self:
            self.arena._batch_transaction = None
        self.active = False

    def abort(self) -> int:
        if not self.active:
            return 0
        if self.committed:
            self.finish_committed()
            return 0
        self.arena.retire()
        pending = [block for state in self.states for block in state.pending_blocks]
        if pending:
            # Verification or a failed copy can still be using destination/scratch
            # pages. No returned page may be reused before its stream completes.
            self.arena.synchronize()
        self.arena.block_manager.release_provisional(pending)
        for state in self.states:
            state.pending_blocks = []
            if state.scratch is not None:
                state.scratch._retired = self.arena._retired
            state._batch_transaction = None
        self.arena._batch_transaction = None
        self.active = False
        return len(pending)
