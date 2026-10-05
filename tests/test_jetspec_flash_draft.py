"""CPU oracle tests for the actual Flash Draft packing/forward adapter.

The injected attention is independent FP32 matmul/softmax, not SDPA. The
reference is the unchanged, tiny official DFlash head and serial drafter.
These tests certify semantics/layout/ownership, not external CUDA execution.
"""
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from transformers import DynamicCache, Qwen3Config

from jetspec.draft_head_adapter import DraftHeadTreeDrafter
from jetspec.models.draft_head import DFlashDraftModel
from nanovllm.speculative.jetspec.flash_draft import FlashDraftProposer, _load_flash_varlen


class VarlenOracle:
    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, cu_q, cu_k, max_q, max_k, *, dropout_p, softmax_scale, causal):
        self.calls.append({"q_offsets": cu_q.tolist(), "k_offsets": cu_k.tolist(),
                           "q_shape": tuple(q.shape), "k_shape": tuple(k.shape),
                           "causal": causal, "max_q": max_q, "max_k": max_k})
        assert cu_q.dtype == cu_k.dtype == torch.int32
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert dropout_p == 0.0
        outputs = []
        for i in range(cu_q.numel() - 1):
            qr = q[cu_q[i]:cu_q[i + 1]].transpose(0, 1).float()
            kr = k[cu_k[i]:cu_k[i + 1]].transpose(0, 1).float()
            vr = v[cu_k[i]:cu_k[i + 1]].transpose(0, 1).float()
            groups = qr.shape[0] // kr.shape[0]
            kr, vr = kr.repeat_interleave(groups, dim=0), vr.repeat_interleave(groups, dim=0)
            scores = (qr @ kr.transpose(-2, -1)) * softmax_scale
            if causal:
                qp = torch.arange(qr.shape[1]) + kr.shape[1] - qr.shape[1]
                allowed = torch.arange(kr.shape[1])[None, :] <= qp[:, None]
                scores.masked_fill_(~allowed, float("-inf"))
            outputs.append((scores.softmax(-1) @ vr).transpose(0, 1).to(q.dtype))
        return torch.cat(outputs)


class FlashDraftTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(481)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        config = Qwen3Config(hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, num_target_layers=6, block_size=4,
            max_position_embeddings=512, attention_dropout=0.0,
            dflash_config={"target_layer_ids": [1, 3], "mask_token_id": 31, "causal_head": True})
        config._attn_implementation = "sdpa"
        self.head = DFlashDraftModel(config).eval()
        self.target = SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(32, 16)),
                                      lm_head=nn.Linear(16, 32, bias=False))
        self.oracle = VarlenOracle()
        self.proposer = FlashDraftProposer(self.head, self.target, flash_varlen=self.oracle)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def request(self, length, *, shift=False):
        return SimpleNamespace(state=SimpleNamespace(
            committed=torch.randint(0, 31, (1, length + 1)),
            target_hidden=torch.randn(1, length, 32)),
            drafter=DraftHeadTreeDrafter(self.head, self.target, 4, [1, 3], shift))

    def copy_request(self, request):
        result = self.request(request.state.target_hidden.shape[1], shift=request.drafter.draft_shift)
        result.state.committed = request.state.committed.clone()
        result.state.target_hidden = request.state.target_hidden.clone()
        result.drafter._fwd.cache = DynamicCache.from_legacy_cache(tuple(
            (k.clone(), v.clone()) for k, v in request.drafter._fwd.cache))
        return result

    def grow(self, request, count):
        request.state.target_hidden = torch.cat((request.state.target_hidden, torch.randn(1, count, 32)), dim=1)
        request.state.committed = torch.cat((request.state.committed, torch.randint(0, 31, (1, count))), dim=1)

    @staticmethod
    def serial(request, depth=3):
        return request.drafter.propose_logits(request.state.committed, depth,
                                             target_hidden=request.state.target_hidden)

    def compare(self, requests, depth=3):
        refs = [self.copy_request(request) for request in requests]
        expected = [self.serial(request, depth) for request in refs]
        actual = self.proposer.propose(requests, depth)
        for request, ref, got, want in zip(requests, refs, actual, expected):
            torch.testing.assert_close(got, want, atol=3e-6, rtol=3e-5)
            cache = request.drafter._fwd.cache
            self.assertEqual(cache.get_seq_length(), request.state.target_hidden.shape[1])
            for (keys, values), (ref_k, ref_v) in zip(cache, ref.drafter._fwd.cache):
                torch.testing.assert_close(keys, ref_k, atol=3e-6, rtol=3e-5)
                torch.testing.assert_close(values, ref_v, atol=3e-6, rtol=3e-5)
                for tensor in (keys, values):
                    self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())
        return actual

    def test_cold_ragged_causal_packs_only_real_context_and_noise(self):
        self.compare([self.request(length) for length in (3, 5, 4)])
        self.assertEqual(len(self.oracle.calls), 2)
        call = self.oracle.calls[0]
        self.assertEqual(call["q_offsets"], [0, 4, 8, 12])
        self.assertEqual(call["k_offsets"], [0, 7, 15, 24])
        self.assertEqual(call["k_shape"], (24, 2, 4))
        self.assertTrue(call["causal"])
        stats = self.proposer.last_stats
        self.assertEqual(stats["batch_sizes"], [3])
        self.assertEqual(stats["attention_key_padding_slots"], 0)
        self.assertEqual(stats["serial_forward_calls"], 0)

    def test_singleton_uses_the_same_real_varlen_api(self):
        self.compare([self.request(8)])
        self.assertEqual(len(self.oracle.calls), 2)
        self.assertEqual(self.proposer.last_stats["singleton_forward_calls"], 1)
        self.assertEqual(self.proposer.last_stats["flash_attention_calls"], 2)
        self.assertEqual(self.proposer.last_stats["serial_forward_calls"], 0)

    def test_warm_ragged_old_context_rope_and_growing_suffix(self):
        requests = [self.request(length) for length in (3, 5, 4)]
        self.proposer.propose(requests)
        history = [[(k.clone(), v.clone()) for k, v in req.drafter._fwd.cache] for req in requests]
        for request, count in zip(requests, (1, 3, 2)):
            self.grow(request, count)
        self.compare(requests)
        for request, old in zip(requests, history):
            for (k, v), (old_k, old_v) in zip(request.drafter._fwd.cache, old):
                self.assertTrue(torch.equal(k[:, :, :old_k.shape[-2]], old_k))
                self.assertTrue(torch.equal(v[:, :, :old_v.shape[-2]], old_v))

    def test_bidirectional_block_and_both_logit_slices_match_official(self):
        self.head.causal_head = False
        for shift in (False, True):
            self.compare([self.request(3, shift=shift), self.request(5, shift=shift)], depth=2)
        self.assertTrue(all(not call["causal"] for call in self.oracle.calls))

    def test_repeat_without_new_context_does_not_persist_noise(self):
        requests = [self.request(3), self.request(5)]
        self.proposer.propose(requests)
        old = [[(k.clone(), v.clone()) for k, v in req.drafter._fwd.cache] for req in requests]
        self.compare(requests)
        for request, history in zip(requests, old):
            for (k, v), (old_k, old_v) in zip(request.drafter._fwd.cache, history):
                self.assertTrue(torch.equal(k, old_k))
                self.assertTrue(torch.equal(v, old_v))

    def test_zero_context_and_stale_longer_cache_reset(self):
        first, second = self.request(8), self.request(4)
        self.serial(first)
        self.serial(second)
        first.state.target_hidden = first.state.target_hidden[:, :0].clone()
        first.state.committed = first.state.committed[:, :1].clone()
        self.compare([first, second])
        self.assertEqual(first.drafter._fwd.cache.get_seq_length(), 0)

    def test_same_shape_other_request_mutation_is_bitwise_isolated(self):
        requests = [self.request(3), self.request(4), self.request(5)]
        self.proposer.propose(requests)
        for request in requests:
            self.grow(request, 2)
        refs = [self.copy_request(request) for request in requests]
        changed = [self.copy_request(request) for request in requests]
        for request in changed[1:]:
            request.state.target_hidden.add_(7)
            request.state.committed[:, -1].add_(1).remainder_(31)
            for k, v in request.drafter._fwd.cache:
                k.add_(2)
                v.sub_(3)
        expected = self.proposer.propose(refs)
        actual = self.proposer.propose(changed)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        for (k, v), (rk, rv) in zip(changed[0].drafter._fwd.cache, refs[0].drafter._fwd.cache):
            self.assertTrue(torch.equal(k, rk))
            self.assertTrue(torch.equal(v, rv))
        self.assertFalse(torch.equal(expected[1], actual[1]))

    def test_poisoned_projection_padding_never_enters_attention_or_cache(self):
        requests = [self.request(3), self.request(5)]
        refs = [self.copy_request(request) for request in requests]
        expected = self.proposer.propose(refs)
        def poison(module, args):
            args[0][0, 3:] = 1234
        hook = self.head.fc.register_forward_pre_hook(poison)
        try:
            actual = self.proposer.propose(requests)
        finally:
            hook.remove()
        for want, got in zip(expected, actual):
            self.assertTrue(torch.equal(want, got))

    def test_failure_in_later_group_publishes_no_cache(self):
        requests = [self.request(3), self.request(4), self.request(128)]
        old = [req.drafter._fwd.cache for req in requests]
        calls = 0
        def fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("later Flash group failed")
            return self.oracle(*args, **kwargs)
        self.proposer.flash_varlen = fail
        with self.assertRaisesRegex(RuntimeError, "later Flash group"):
            self.proposer.propose(requests)
        for request, cache in zip(requests, old):
            self.assertIs(request.drafter._fwd.cache, cache)
            self.assertEqual(cache.get_seq_length(), 0)

    def test_failed_lm_head_preserves_warm_cache_objects_and_bytes(self):
        requests = [self.request(3), self.request(4)]
        self.proposer.propose(requests)
        old = [req.drafter._fwd.cache for req in requests]
        history = [[(k.clone(), v.clone()) for k, v in cache] for cache in old]
        for req in requests:
            self.grow(req, 1)
        with patch.object(self.target.lm_head, "forward", side_effect=RuntimeError("lm head failed")):
            with self.assertRaisesRegex(RuntimeError, "lm head failed"):
                self.proposer.propose(requests)
        for request, cache, before in zip(requests, old, history):
            self.assertIs(request.drafter._fwd.cache, cache)
            for (k, v), (bk, bv) in zip(cache, before):
                self.assertTrue(torch.equal(k, bk))
                self.assertTrue(torch.equal(v, bv))

    def test_reordered_survivor_and_newcomer_preserve_request_owned_cache(self):
        requests = [self.request(4), self.request(5), self.request(6)]
        self.proposer.propose(requests)
        self.grow(requests[2], 2)
        self.compare([requests[2], self.request(3)])
        self.assertFalse(hasattr(self.proposer, "requests"))

    def test_projection_grouping_uses_flash_even_for_outlier_singleton(self):
        warm = [self.request(12), self.request(15)]
        for req in warm:
            self.serial(req)
            self.grow(req, 1)
        self.compare(warm + [self.request(128)])
        stats = self.proposer.last_stats
        self.assertEqual(stats["batch_sizes"], [2, 1])
        self.assertEqual(stats["batched_forward_calls"], 1)
        self.assertEqual(stats["singleton_forward_calls"], 1)
        self.assertEqual(stats["serial_forward_calls"], 0)

    def test_unsupported_head_fails_instead_of_silent_sdpa_fallback(self):
        request = self.request(3)
        for variant in ("training", "dynamic", "custom"):
            saved_rope = self.head.rotary_emb.rope_type
            saved_drafter = request.drafter
            if variant == "training":
                self.head.train()
            elif variant == "dynamic":
                self.head.rotary_emb.rope_type = "dynamic"
            else:
                request.drafter = Mock()
            try:
                with self.assertRaisesRegex(RuntimeError, "supported official"):
                    self.proposer.propose([request])
            finally:
                self.head.eval()
                self.head.rotary_emb.rope_type = saved_rope
                request.drafter = saved_drafter

    def test_adapter_does_not_mutate_head_weights_or_attention_backend(self):
        old = {name: value.clone() for name, value in self.head.state_dict().items()}
        self.proposer.propose([self.request(3), self.request(4)])
        for name, value in self.head.state_dict().items():
            self.assertTrue(torch.equal(value, old[name]))
        self.assertEqual(self.head.config._attn_implementation, "sdpa")

    def test_validation_and_missing_or_old_dependency_are_explicit(self):
        self.assertEqual(self.proposer.propose([]), [])
        req = self.request(3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.proposer.propose([req, req])
        for depth in (0, 4, True, 1.5):
            with self.assertRaises(ValueError):
                self.proposer.propose([req], depth)
        with patch.dict(sys.modules, {"flash_attn": None}):
            with self.assertRaisesRegex(RuntimeError, "external flash-attn"):
                _load_flash_varlen()
        old_module = SimpleNamespace(__version__="2.0.0", flash_attn_varlen_func=Mock())
        with patch.dict(sys.modules, {"flash_attn": old_module}):
            with self.assertRaisesRegex(RuntimeError, ">= 2.1"):
                _load_flash_varlen()


if __name__ == "__main__":
    unittest.main()
