from __future__ import annotations

import time
from typing import Any

import torch

from nanovllm.speculative.jetspec.drafter import load_official_drafter
from nanovllm.speculative.jetspec.state import DenseTargetState
from nanovllm.utils.context import reset_context


class JetSpecRuntime:
    """Single-request, greedy, eager JetSpec correctness MVP.

    Tree construction and acceptance reuse engine-independent code from pinned
    JetSpec commit 2c7b3fa. Target execution uses nano-vLLM Qwen3 weights and an
    opt-in dense SDPA tree mask. No Scheduler, prefix cache, CUDA graph or TP path
    is involved.
    """

    def __init__(self, target, tokenizer, draft_model: str, *, tree_depth: int = 15,
                 tree_width: int = 7, tree_budget: int = 63):
        if tree_depth != 15 or tree_width != 7 or tree_budget != 63:
            raise ValueError("Phase-1 MVP is fixed to depth=15, width=7, budget=63")
        self.target = target
        self.tokenizer = tokenizer
        self.tree_depth = int(tree_depth)
        self.tree_width = int(tree_width)
        self.tree_budget = int(tree_budget)
        self.head, self.drafter = load_official_drafter(draft_model, target, tree_depth)
        self.target_layer_ids = tuple(int(i) for i in self.head.target_layer_ids)
        if self.target_layer_ids != (1, 9, 17, 25, 33):
            raise ValueError(f"unexpected target taps: {self.target_layer_ids}")
        from jetspec.tree import get_algorithm

        self.tree_algorithm = get_algorithm("accum_logp")
        self.eos_token_ids = self._resolve_eos()
        self._active_state: DenseTargetState | None = None

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
                 return_rounds: bool = True) -> dict[str, Any]:
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
        state = DenseTargetState(committed, prompt_kv, prompt_hidden.unsqueeze(0))
        self._active_state = state
        output_ids = [int(first.item())]
        rounds = []

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
            allowed = torch.zeros((n, past_len + n), dtype=torch.bool, device=device)
            allowed[:, :past_len] = True
            allowed[:, past_len:] = ancestor
            positions = past_len + tree.depth.long()
            target_logits, provisional_kv, node_hidden = self._target_forward(
                tree.token_ids,
                positions,
                state.key_values,
                allowed,
                True,
            )
            greedy = target_logits.argmax(dim=-1)
            path, accepted_len, correction = gpu_tree_accept(
                tree.token_ids,
                greedy,
                tree.parent_indices,
                tree.depth,
                max_depth=self.tree_depth,
            )
            state.commit_tree_path(provisional_kv, node_hidden.unsqueeze(0), path)
            accepted = tree.token_ids.index_select(0, path[1:])
            block = torch.cat((accepted, correction.view(1)))
            state.committed = torch.cat((state.committed, block.view(1, -1)), dim=1)
            state.assert_round_invariant()

            round_record = {
                "tree_size": n,
                "accepted_draft_length": int(accepted_len),
                "accepted_length_including_correction": int(accepted_len) + 1,
                "accepted_path_node_indices": [int(x) for x in path.tolist()],
                "accepted_path_token_ids_root_inclusive": [
                    int(x) for x in tree.token_ids.index_select(0, path).tolist()
                ],
                "correction_token_id": int(correction.item()),
            }
            if return_rounds:
                rounds.append(round_record)
            for value in block.tolist():
                output_ids.append(int(value))
                if int(value) in self.eos_token_ids:
                    break

        output_ids = output_ids[:max_new_tokens]
        torch.cuda.synchronize()
        latency = time.perf_counter() - start
        final_invariant = {
            "committed_minus_one": int(state.committed.shape[1]) - 1,
            "target_kv_length": state.cache_len,
            "target_feature_length": int(state.target_hidden.shape[1]),
            "draft_cache_length": int(self.drafter._fwd.cache.get_seq_length()),
        }
        final_invariant["draft_pending_committed_feature_suffix"] = (
            final_invariant["target_feature_length"] - final_invariant["draft_cache_length"]
        )
        clean = (
            final_invariant["committed_minus_one"]
            == final_invariant["target_kv_length"]
            == final_invariant["target_feature_length"]
            and 0 <= final_invariant["draft_cache_length"] <= final_invariant["target_feature_length"]
        )
        # No generation state is reusable accidentally. A later call starts from empty
        # target state and reset Draft KV; rejected provisional tensors have no owner.
        state.clear()
        self.drafter.reset_cache()
        self._active_state = None
        return {
            "token_ids": output_ids,
            "text": self.tokenizer.decode(output_ids, skip_special_tokens=True),
            "rounds": rounds,
            "target_verification_rounds": len(rounds),
            "accept_lengths": [r["accepted_length_including_correction"] for r in rounds],
            "tree_sizes": [r["tree_size"] for r in rounds],
            "latency_s": latency,
            "state_invariant_before_cleanup": final_invariant,
            "state_invariant_passed": clean,
            "stale_speculative_state_after_cleanup": self._active_state is not None,
        }
