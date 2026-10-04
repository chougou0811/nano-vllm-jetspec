"""CPU qualification of genuine batched official DFlash Draft forwards.

Uses a tiny randomly initialized official head, not a replica forward or model
checkpoint. Production batched logits, masks, RoPE and cache publication are
compared against the original per-request DraftHeadTreeDrafter.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from transformers import DynamicCache, Qwen3Config

from jetspec.draft_head_adapter import DraftHeadTreeDrafter
from jetspec.models.draft_head import DFlashDraftModel
from nanovllm.speculative.jetspec.batched_draft import BatchedDraftProposer


class BatchedDraftTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(481)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        config = Qwen3Config(hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, num_target_layers=6, block_size=4,
            max_position_embeddings=512, attention_dropout=0.0,
            dflash_config={"target_layer_ids": [1, 3], "mask_token_id": 31,
                           "causal_head": True})
        config._attn_implementation = "sdpa"
        self.head = DFlashDraftModel(config).eval()
        self.target = SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(32, 16)),
                                      lm_head=nn.Linear(16, 32, bias=False))
        self.proposer = BatchedDraftProposer(self.head, self.target)

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

    @staticmethod
    def serial(request, depth=3):
        return request.drafter.propose_logits(request.state.committed, depth,
                                             target_hidden=request.state.target_hidden)

    def compare_caches(self, requests, references, *, compact_indices=None):
        compact_indices = set(range(len(requests))) if compact_indices is None else set(compact_indices)
        for index, (request, reference) in enumerate(zip(requests, references)):
            cache = request.drafter._fwd.cache
            self.assertEqual(cache.get_seq_length(), request.state.target_hidden.shape[1])
            self.assertEqual(len(cache), 2)
            for (keys, values), (ref_k, ref_v) in zip(cache, reference.drafter._fwd.cache):
                torch.testing.assert_close(keys, ref_k, atol=3e-6, rtol=3e-5)
                torch.testing.assert_close(values, ref_v, atol=3e-6, rtol=3e-5)
                for tensor in (keys, values):
                    self.assertEqual(tensor.shape[0], 1)
                    if index in compact_indices:
                        self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size(),
                                         "a persistent row retained batch/padding storage")

    def compare_serial(self, requests, *, depth=3, compact_indices=None):
        refs = [self.copy_request(request) for request in requests]
        expected = [self.serial(request, depth) for request in refs]
        actual = self.proposer.propose(requests, depth)
        for got, ref in zip(actual, expected):
            self.assertEqual(got.shape, (1, depth, 32))
            torch.testing.assert_close(got, ref, atol=3e-6, rtol=3e-5)
        self.compare_caches(requests, refs, compact_indices=compact_indices)
        return actual

    def grow(self, request, count):
        request.state.target_hidden = torch.cat((request.state.target_hidden, torch.randn(1, count, 32)), dim=1)
        request.state.committed = torch.cat((request.state.committed, torch.randint(0, 31, (1, count))), dim=1)

    def test_cold_ragged_requests_use_one_real_head_forward(self):
        requests = [self.request(length) for length in (3, 5, 4)]
        shapes = []
        hook = self.head.register_forward_pre_hook(
            lambda module, args, kwargs: shapes.append(kwargs["noise_embedding"].shape[0]), with_kwargs=True)
        try:
            self.compare_serial(requests)
        finally:
            hook.remove()
        self.assertEqual(shapes, [1, 1, 1, 3])
        self.assertEqual(self.proposer.last_stats["batch_sizes"], [3])
        self.assertEqual(self.proposer.last_stats["batched_forward_calls"], 1)
        self.assertEqual(self.proposer.last_stats["serial_forward_calls"], 0)

    def test_warm_ragged_cache_appends_only_new_taps_and_preserves_history(self):
        requests = [self.request(length) for length in (3, 5, 4)]
        self.proposer.propose(requests)
        old = [[(k.clone(), v.clone()) for k, v in request.drafter._fwd.cache] for request in requests]
        for request, count in zip(requests, (1, 3, 2)):
            self.grow(request, count)
        self.compare_serial(requests)
        for request, history in zip(requests, old):
            for (keys, values), (old_k, old_v) in zip(request.drafter._fwd.cache, history):
                self.assertTrue(torch.equal(keys[:, :, :old_k.shape[-2]], old_k))
                self.assertTrue(torch.equal(values[:, :, :old_v.shape[-2]], old_v))
        self.assertEqual(self.proposer.last_stats["batch_sizes"], [3])

    def test_explicit_mask_and_absolute_rope_cover_ragged_old_and_new_keys(self):
        requests = [self.request(2), self.request(5)]
        for request in requests:
            self.serial(request)
        self.grow(requests[0], 3)
        self.grow(requests[1], 1)
        captured = []
        hook = self.head.register_forward_pre_hook(
            lambda module, args, kwargs: captured.append(kwargs), with_kwargs=True)
        try:
            self.proposer.propose(requests)
        finally:
            hook.remove()
        kwargs = captured[0]
        # Group sorting puts suffix-length one before suffix-length three.
        self.assertTrue(torch.equal(kwargs["position_ids"], torch.tensor(
            [[5, 0, 0, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8]])))
        allowed = kwargs["attention_mask"][:, 0] == 0
        expected = torch.zeros(2, 4, 12, dtype=torch.bool)
        expected[0, :, :5] = True
        expected[0, :, 5] = True
        expected[1, :, :2] = True
        expected[1, :, 5:8] = True
        expected[:, :, 8:] = torch.ones(4, 4, dtype=torch.bool).tril()
        self.assertTrue(torch.equal(allowed, expected))
        self.assertIs(kwargs["is_causal"], False)

    def test_bidirectional_head_preserves_official_noncausal_block_visibility(self):
        self.head.causal_head = False
        self.compare_serial([self.request(3), self.request(5)])

    def test_eager_attention_backend_uses_the_same_explicit_ragged_mask(self):
        self.head.config._attn_implementation = "eager"
        self.compare_serial([self.request(3), self.request(5)])

    def test_repeated_proposal_without_new_context_does_not_cache_noise(self):
        requests = [self.request(3), self.request(5)]
        self.proposer.propose(requests)
        old = [[(k.clone(), v.clone()) for k, v in request.drafter._fwd.cache] for request in requests]
        self.compare_serial(requests)
        for request, history in zip(requests, old):
            for (keys, values), (old_k, old_v) in zip(request.drafter._fwd.cache, history):
                self.assertTrue(torch.equal(keys, old_k))
                self.assertTrue(torch.equal(values, old_v))

    def test_shifted_draft_slice_matches_original_adapter(self):
        self.compare_serial([self.request(3, shift=True), self.request(5, shift=True)], depth=2)

    def test_same_shape_other_request_mutation_is_bitwise_isolated(self):
        requests = [self.request(3), self.request(4), self.request(5)]
        self.proposer.propose(requests)
        for request in requests:
            self.grow(request, 2)
        reference = [self.copy_request(request) for request in requests]
        changed = [self.copy_request(request) for request in requests]
        for request in changed[1:]:
            request.state.target_hidden.add_(7)
            request.state.committed[:, -1].add_(1).remainder_(31)
            for keys, values in request.drafter._fwd.cache:
                keys.add_(2)
                values.sub_(3)
        ref = self.proposer.propose(reference)
        actual = self.proposer.propose(changed)
        self.assertTrue(torch.equal(ref[0], actual[0]))
        for (a_k, a_v), (b_k, b_v) in zip(reference[0].drafter._fwd.cache, changed[0].drafter._fwd.cache):
            self.assertTrue(torch.equal(a_k, b_k))
            self.assertTrue(torch.equal(a_v, b_v))
        self.assertFalse(torch.equal(ref[1], actual[1]))

    def test_finite_poisoned_new_suffix_padding_is_never_visible_or_cached(self):
        requests = [self.request(3), self.request(5)]
        refs = [self.copy_request(request) for request in requests]
        ref = self.proposer.propose(refs)

        def poison(module, args, kwargs):
            kwargs["target_hidden"][0, 3:] = 1234
            kwargs["position_ids"][0, 3:5] = 237

        hook = self.head.register_forward_pre_hook(poison, with_kwargs=True)
        try:
            actual = self.proposer.propose(requests)
        finally:
            hook.remove()
        for got, expected in zip(actual, ref):
            self.assertTrue(torch.equal(got, expected))
        self.compare_caches(requests, refs)

    def test_failed_head_forward_preserves_all_persistent_cache_objects_and_bytes(self):
        requests = [self.request(3), self.request(4)]
        self.proposer.propose(requests)
        for request in requests:
            self.grow(request, 1)
        old = [request.drafter._fwd.cache for request in requests]
        tensors = [[(k.clone(), v.clone()) for k, v in cache] for cache in old]
        error = RuntimeError("after head cache updates")

        def fail(module, args, output):
            raise error

        hook = self.head.register_forward_hook(fail)
        try:
            with self.assertRaises(RuntimeError) as caught:
                self.proposer.propose(requests)
            self.assertIs(caught.exception, error)
        finally:
            hook.remove()
        for request, cache, history in zip(requests, old, tensors):
            self.assertIs(request.drafter._fwd.cache, cache)
            for (keys, values), (old_k, old_v) in zip(cache, history):
                self.assertTrue(torch.equal(keys, old_k))
                self.assertTrue(torch.equal(values, old_v))
        self.compare_serial(requests)

    def test_failed_lm_head_does_not_publish_staged_batched_caches(self):
        requests = [self.request(3), self.request(4)]
        old = [request.drafter._fwd.cache for request in requests]
        with patch.object(self.target.lm_head, "forward", side_effect=RuntimeError("lm head failed")):
            with self.assertRaisesRegex(RuntimeError, "lm head failed"):
                self.proposer.propose(requests)
        for request, cache in zip(requests, old):
            self.assertIs(request.drafter._fwd.cache, cache)
            self.assertEqual(cache.get_seq_length(), 0)

    def test_stale_longer_cache_is_reset_without_changing_other_requests(self):
        first, second = self.request(8), self.request(4)
        self.serial(first)
        self.serial(second)
        first.state.target_hidden = first.state.target_hidden[:, :3].clone()
        first.state.committed = first.state.committed[:, :4].clone()
        self.compare_serial([first, second])
        self.assertEqual(first.drafter._fwd.cache.get_seq_length(), 3)

    def test_reordered_survivor_and_new_arrival_keep_request_owned_history(self):
        requests = [self.request(4), self.request(5), self.request(6)]
        self.proposer.propose(requests)
        survivor = requests[2]
        self.grow(survivor, 2)
        newcomer = self.request(3)
        self.compare_serial([survivor, newcomer])
        self.assertFalse(hasattr(self.proposer, "requests"), "proposer retained cancelled requests")

    def test_grouping_bounds_cold_suffix_padding_and_counts_serial_fallback(self):
        warm = [self.request(12), self.request(15)]
        for request in warm:
            self.serial(request)
            self.grow(request, 1)
        requests = warm + [self.request(128)]
        self.compare_serial(requests, compact_indices=(0, 1))
        stats = self.proposer.last_stats
        self.assertEqual(stats["batch_sizes"], [2])
        self.assertEqual(stats["serial_forward_calls"], 1)
        self.assertLessEqual(stats["max_padding_ratio"], 2)
        self.assertGreater(stats["cache_bytes_after"], stats["cache_bytes_before"])

    def test_disabled_ablation_has_only_original_serial_head_forwards(self):
        self.proposer.enabled = False
        requests = [self.request(3), self.request(4)]
        self.compare_serial(requests, compact_indices=())
        self.assertEqual(self.proposer.last_stats["batch_sizes"], [])
        self.assertEqual(self.proposer.last_stats["serial_forward_calls"], 2)
        stats = self.proposer.last_stats
        self.assertGreater(stats["cache_storage_bytes_after"], stats["cache_bytes_after"])

    def test_custom_or_mock_drafter_is_honest_serial_fallback(self):
        requests = [self.request(3), self.request(4)]
        for request in requests:
            request.drafter = Mock()
            request.drafter.propose_logits.return_value = torch.zeros(1, 3, 32)
        output = self.proposer.propose(requests)
        self.assertEqual(len(output), 2)
        self.assertEqual(self.proposer.last_stats["serial_forward_calls"], 2)
        self.assertEqual(self.proposer.last_stats["batched_forward_calls"], 0)
        for request in requests:
            request.drafter.propose_logits.assert_called_once()

    def test_dynamic_rope_or_training_head_is_not_silently_adapted(self):
        # Dynamic RoPE updates frequencies from the batch-global maximum. That
        # is not a request-independent positional transform, so use the original
        # proposer rather than advertising an unsupported isolated batched path.
        for variant in ("dynamic", "longrope", "training"):
            requests = [self.request(3), self.request(4)]
            saved_type = self.head.rotary_emb.rope_type
            if variant == "training":
                self.head.train()
            else:
                self.head.rotary_emb.rope_type = variant
            for request in requests:
                request.drafter.propose_logits = Mock(return_value=torch.zeros(1, 3, 32))
            try:
                self.proposer.propose(requests)
            finally:
                self.head.eval()
                self.head.rotary_emb.rope_type = saved_type
            self.assertEqual(self.proposer.last_stats["batched_forward_calls"], 0)
            self.assertEqual(self.proposer.last_stats["serial_forward_calls"], 2)

    def test_empty_inputs_duplicate_requests_and_invalid_options(self):
        self.assertEqual(self.proposer.propose([]), [])
        request = self.request(3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.proposer.propose([request, request])
        for depth in (0, 4, True, 1.5):
            with self.assertRaises(ValueError):
                self.proposer.propose([request], depth)
        for ratio in (0.5, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                BatchedDraftProposer(self.head, self.target, max_padding_ratio=ratio)
        request.state.committed = request.state.committed[:, :-1]
        with self.assertRaisesRegex(ValueError, "uncached anchor"):
            self.proposer.propose([request])


if __name__ == "__main__":
    unittest.main()
