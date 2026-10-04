"""Incremental prefill ownership/failure and actual tiny-Qwen CPU model gates."""
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import Qwen3Config

from nanovllm.engine.block_manager import BlockManager
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.speculative.jetspec.prefill import offset_causal_mask
from nanovllm.speculative.jetspec.state import TreeScratchArena
from tests.test_jetspec_continuous import make_serving_runtime


def pool_rows(runtime, slots):
    return runtime.kv_pool[:, :, slots // runtime.block_size, slots % runtime.block_size].clone()


class ChunkedPrefillRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.runtime = make_serving_runtime()

    def tearDown(self):
        self.runtime.close()

    def finish_chunks(self, context, chunks):
        ready = None
        for size in chunks:
            ready = self.runtime.prefill_step(context, size)
        return ready

    def test_begin_allocates_no_pages_and_only_last_chunk_promotes(self):
        context = self.runtime.begin_prefill([1, 2, 3, 4], request_id="a")
        self.assertEqual(context.processed_tokens, 0)
        self.assertEqual(context.total_tokens, 4)
        self.assertEqual(context.owned_blocks, [])
        self.assertFalse(self.runtime.requests)
        self.assertIsNone(self.runtime.prefill_step(context, 2))
        self.assertEqual(context.processed_tokens, 2)
        blocks = list(context.owned_blocks)
        ptr = context.feature_storage.data_ptr()
        prefix = context.target_hidden.clone()
        request = self.runtime.prefill_step(context, 10)
        self.assertTrue(context.promoted)
        self.assertFalse(self.runtime.prefills)
        self.assertEqual(request.output_ids, [5])
        self.assertEqual(request.state.owned_blocks, blocks)
        self.assertEqual(request.state._feature_storage.data_ptr(), ptr)
        self.assertTrue(torch.equal(request.state.target_hidden[:, :2], prefix))
        self.assertEqual(request.state.feature_storage_snapshot()["feature_history_copy_bytes"], 0)
        request.state.assert_round_invariant(validate_device=True)
        with self.assertRaises(ValueError):
            self.runtime.prefill_step(context, 1)

    def test_chunks_have_absolute_positions_and_offset_visibility(self):
        dense = self.runtime.target.model.forward_dense
        seen = []
        def observe(ids, positions, past, mask, taps):
            seen.append((positions.tolist(), None if past is None else past[0][0].shape[0],
                         None if mask is None else mask.clone()))
            return dense(ids, positions, past, mask, taps)
        self.runtime.target.model.forward_dense = observe
        context = self.runtime.begin_prefill([1, 2, 3, 4, 5])
        self.finish_chunks(context, [2, 2, 1])
        self.assertEqual([r[0] for r in seen], [[0, 1], [2, 3], [4]])
        self.assertEqual([r[1] for r in seen], [None, 2, 4])
        self.assertIsNone(seen[0][2])
        self.assertTrue(torch.equal(seen[1][2], torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)))
        self.assertTrue(bool(seen[2][2].all()))

    def test_interleaved_requests_have_independent_cursor_pages_and_features(self):
        a = self.runtime.begin_prefill([1, 2, 3], request_id="a")
        b = self.runtime.begin_prefill([11, 12, 13, 14], request_id="b")
        self.runtime.prefill_step(a, 1)
        self.runtime.prefill_step(b, 2)
        self.assertFalse(set(a.owned_blocks) & set(b.owned_blocks))
        a_rows = pool_rows(self.runtime, a.logical_slots)
        self.runtime.prefill_step(b, 2)
        self.assertTrue(torch.equal(pool_rows(self.runtime, a.logical_slots), a_rows))
        ra = self.runtime.prefill_step(a, 2)
        self.assertEqual(ra.output_ids, [4])
        self.assertEqual(self.runtime.requests["b"].output_ids, [15])

    def test_incremental_capacity_crosses_page_boundary_not_whole_prefix(self):
        context = self.runtime.begin_prefill([1] * 257)
        self.assertEqual(self.runtime.estimate_prefill_chunk_capacity(context, 255)["required_free_blocks"], 1)
        self.runtime.prefill_step(context, 255)
        self.assertEqual(self.runtime.estimate_prefill_chunk_capacity(context, 1)["required_free_blocks"], 0)
        self.runtime.prefill_step(context, 1)
        self.assertEqual(self.runtime.estimate_prefill_chunk_capacity(context, 1)["required_free_blocks"], 1)
        request = self.runtime.prefill_step(context, 1)
        self.assertEqual(len(request.state.owned_blocks), 2)

    def test_allocator_backpressure_preserves_existing_partial_for_retry(self):
        runtime = self.runtime
        context = runtime.begin_prefill([1] * 257)
        runtime.prefill_step(context, 256)
        old = pool_rows(runtime, context.logical_slots)
        held = runtime.block_manager.reserve_provisional(len(runtime.block_manager.free_block_ids))
        self.assertFalse(runtime.estimate_prefill_chunk_capacity(context, 1)["feasible"])
        with self.assertRaisesRegex(RuntimeError, "insufficient KV"):
            runtime.prefill_step(context, 1)
        self.assertIs(runtime.prefills[context.request_id], context)
        self.assertEqual(context.processed_tokens, 256)
        self.assertTrue(torch.equal(pool_rows(runtime, context.logical_slots), old))
        runtime.block_manager.release_provisional(held)
        self.assertIsNotNone(runtime.prefill_step(context, 1))

    def test_forward_after_partial_write_failure_releases_all_pages(self):
        context = self.runtime.begin_prefill([1, 2, 3])
        self.runtime.prefill_step(context, 1)
        def fail(*args):
            self.runtime.kv_pool.fill_(17)
            raise RuntimeError("partial layer write")
        self.runtime.target.model.forward_dense = fail
        with self.assertRaisesRegex(RuntimeError, "partial layer write"):
            self.runtime.prefill_step(context, 1)
        self.assertTrue(context.cancelled)
        self.assertFalse(self.runtime.prefills)
        self.assertEqual(self.runtime.block_manager.used_block_ids, set(self.runtime.arena.blocks))
        self.assertIn("partial layer write", context.error)

    def test_head_promotion_failure_releases_and_never_publishes_ready(self):
        context = self.runtime.begin_prefill([1, 2])
        self.runtime.target.lm_head = Mock(side_effect=RuntimeError("head failed"))
        with self.assertRaisesRegex(RuntimeError, "head failed"):
            self.runtime.prefill_step(context, 2)
        self.assertFalse(self.runtime.requests)
        self.assertFalse(self.runtime.prefills)
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_drafter_construction_failure_releases_without_double_ownership(self):
        context = self.runtime.begin_prefill([1, 2])
        self.runtime._new_drafter = Mock(side_effect=RuntimeError("drafter failed"))
        with self.assertRaisesRegex(RuntimeError, "drafter failed"):
            self.runtime.prefill_step(context, 2)
        self.assertFalse(self.runtime.block_manager.used_block_ids)
        self.assertTrue(context.cancelled)

    def test_wrong_tap_geometry_cleans_partial(self):
        dense = self.runtime.target.model.forward_dense
        def malformed(*args):
            hidden, kv, taps = dense(*args)
            return hidden, kv, taps[:-1]
        self.runtime.target.model.forward_dense = malformed
        context = self.runtime.begin_prefill([1, 2])
        with self.assertRaisesRegex(ValueError, "prefill taps"):
            self.runtime.prefill_step(context, 2)
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_cancel_partial_is_idempotent_and_id_can_be_reused(self):
        context = self.runtime.begin_prefill([1, 2], request_id="same")
        self.runtime.prefill_step(context, 1)
        self.assertEqual(self.runtime.cancel_prefill(context), 1)
        self.assertEqual(self.runtime.cancel_prefill(context), 0)
        replacement = self.runtime.begin_prefill([3], request_id="same")
        self.assertIsNot(context, replacement)

    def test_close_releases_both_partial_and_ready(self):
        partial = self.runtime.begin_prefill([1, 2], request_id="partial")
        ready = self.runtime.begin_prefill([3], request_id="ready")
        self.runtime.prefill_step(partial, 1)
        self.runtime.prefill_step(ready, 1)
        snapshot = self.runtime.capacity_snapshot()
        self.assertEqual(snapshot["prefill_requests"], 1)
        self.assertEqual(snapshot["prefill_blocks"], 1)
        self.runtime.close()
        self.assertFalse(self.runtime.requests)
        self.assertFalse(self.runtime.prefills)
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_resume_preserves_anchor_outputs_rounds_and_preemption_count(self):
        original = self.runtime.create_request([1, 2, 3], request_id="resume", max_new_tokens=8)
        self.runtime.step([original], tree_budgets=[1], record_timing=False)
        snapshot = self.runtime.suspend(original)
        context = self.runtime.begin_prefill(snapshot=snapshot)
        self.assertEqual(context.total_tokens, len(snapshot["committed_tokens"]) - 1)
        self.runtime.target.lm_head = Mock(side_effect=AssertionError("resume cannot repredict anchor"))
        request = self.finish_chunks(context, [1] * context.total_tokens)
        self.assertEqual(request.output_ids, snapshot["output_ids"])
        self.assertEqual(request.state.committed[0].tolist(), snapshot["committed_tokens"])
        self.assertEqual(request.rounds, snapshot["rounds"])
        self.assertEqual(request.preemptions, 1)
        self.assertEqual(request.prompt_length, 3)

    def test_partial_blocks_policy_changes_and_duplicate_full_prefill_ids(self):
        context = self.runtime.begin_prefill([1, 2], request_id="a")
        with self.assertRaises(RuntimeError):
            self.runtime.configure_optimizations(feature_storage=True)
        with self.assertRaises(ValueError):
            self.runtime.create_request([1], request_id="a")
        other = self.runtime.create_request([1])
        self.assertNotEqual(other.request_id, context.request_id)

    def test_optimized_features_transfer_existing_buffer_without_reenable(self):
        self.runtime.configure_optimizations(lightweight=True, feature_storage=True)
        context = self.runtime.begin_prefill([1, 2, 3])
        self.runtime.prefill_step(context, 1)
        ptr = context.feature_storage.data_ptr()
        request = self.runtime.prefill_step(context, 2)
        self.assertEqual(request.state._feature_storage.data_ptr(), ptr)
        self.assertFalse(request.state.validate_device)
        request.state.assert_round_invariant(validate_device=True)

    def test_manual_batch_wrapper_rejects_partial_before_any_cuda_operation(self):
        context = self.runtime.begin_prefill([1, 2])
        with patch("torch.cuda.synchronize", side_effect=AssertionError("must reject before CUDA")):
            with self.assertRaisesRegex(RuntimeError, "no existing live"):
                self.runtime.generate_batch([[3]])
        self.assertIs(self.runtime.prefills[context.request_id], context)

    def test_foreign_context_and_invalid_chunk_inputs_do_not_mutate(self):
        context = self.runtime.begin_prefill([1, 2])
        other = make_serving_runtime()
        try:
            with self.assertRaises(ValueError):
                other.prefill_step(context, 1)
            for count in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    self.runtime.prefill_step(context, count)
            self.assertEqual(context.owned_blocks, [])
        finally:
            other.close()

    def test_snapshot_rejects_bad_prompt_length_or_output_history(self):
        original = self.runtime.create_request([1, 2, 3], max_new_tokens=8)
        self.runtime.step([original], tree_budgets=[1], record_timing=False)
        snapshot = self.runtime.suspend(original)
        for changed in ({"prompt_length": 2}, {"output_ids": [99, snapshot["output_ids"][-1]]}):
            malformed = dict(snapshot, **changed)
            with self.assertRaisesRegex(ValueError, "invalid suspended"):
                self.runtime.begin_prefill(snapshot=malformed)
        self.assertFalse(self.runtime.prefills)
        self.assertEqual(self.runtime.block_manager.used_block_ids, set(self.runtime.arena.blocks))

    def test_failed_release_retains_ownership_for_explicit_cleanup_retry(self):
        context = self.runtime.begin_prefill([1, 2])
        self.runtime.prefill_step(context, 1)
        with patch.object(self.runtime.block_manager, "release_provisional", side_effect=RuntimeError("release failed")):
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                self.runtime.cancel_prefill(context)
        self.assertIs(self.runtime.prefills[context.request_id], context)
        self.assertFalse(context.cancelled)
        self.assertEqual(len(context.owned_blocks), 1)
        self.assertEqual(self.runtime.cancel_prefill(context), 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class ChunkedPrefillStreamTest(unittest.TestCase):
    def setUp(self):
        self.runtime = make_serving_runtime()
        self.runtime.kv_pool = self.runtime.kv_pool.cuda()
        self.runtime.arena = TreeScratchArena(self.runtime.kv_pool, self.runtime.block_manager, 256)
        def dense(ids, positions, past, mask, taps):
            hidden = torch.zeros(ids.numel(), 6, device=ids.device)
            hidden[:, 0] = ids
            values = ids.float()[:, None, None].expand(-1, 2, 4).clone()
            kv = [(values + layer, values + layer + 1) for layer in range(3)]
            return hidden, kv, hidden.clone()
        def head(hidden):
            logits = torch.zeros(hidden.shape[0], 64, device=hidden.device)
            logits[torch.arange(hidden.shape[0], device=hidden.device), hidden[:, 0].long() + 1] = 1
            return logits
        self.runtime.target.model.forward_dense = dense
        self.runtime.target.lm_head = head

    def tearDown(self):
        self.runtime.close()
        torch.cuda.synchronize()

    def delayed_record(self, context):
        original = context.record_ready
        def record():
            if hasattr(torch.cuda, "_sleep"):
                torch.cuda._sleep(20_000_000)
            return original()
        return patch.object(context, "record_ready", side_effect=record)

    def test_chunks_cross_streams_then_promote_without_stale_prefix(self):
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            context = self.runtime.begin_prefill([1, 2, 3])
            with self.delayed_record(context):
                self.runtime.prefill_step(context, 1)
        with torch.cuda.stream(consumer):
            request = self.runtime.prefill_step(context, 2)
            request.state.assert_round_invariant(validate_device=True)
            self.assertEqual(request.output_ids, [4])
            self.assertEqual(request.state.target_hidden[0, :, 0].tolist(), [1., 2., 3.])
            rows = pool_rows(self.runtime, request.state.logical_slots)
            self.assertTrue(torch.equal(rows[0, 0, :, 0, 0], torch.tensor([1., 2., 3.], device="cuda")))

    def test_cancel_on_other_stream_fences_outstanding_chunk_writes(self):
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            context = self.runtime.begin_prefill([1, 2, 3])
            with self.delayed_record(context):
                self.runtime.prefill_step(context, 1)
            completion = context._ready
            if hasattr(torch.cuda, "_sleep"):
                self.assertFalse(completion.query(), "test should exercise pending producer writes")
        with torch.cuda.stream(consumer):
            self.runtime.cancel_prefill(context)
        self.assertTrue(completion.query())
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_failed_partial_forward_fences_unrecorded_writes_before_release(self):
        producer = torch.cuda.Stream()
        completion = torch.cuda.Event()
        def fail(ids, *args):
            if hasattr(torch.cuda, "_sleep"):
                torch.cuda._sleep(20_000_000)
            self.runtime.kv_pool.fill_(23)
            completion.record(producer)
            raise RuntimeError("queued layer write failed")
        self.runtime.target.model.forward_dense = fail
        with torch.cuda.stream(producer):
            context = self.runtime.begin_prefill([1, 2])
            with self.assertRaisesRegex(RuntimeError, "queued layer write failed"):
                self.runtime.prefill_step(context, 1)
        self.assertTrue(completion.query())
        self.assertFalse(self.runtime.block_manager.used_block_ids)
        self.assertFalse(self.runtime.prefills)

    def test_close_on_other_stream_fences_partial_context(self):
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            context = self.runtime.begin_prefill([1, 2])
            with self.delayed_record(context):
                self.runtime.prefill_step(context, 1)
            completion = context._ready
        with torch.cuda.stream(consumer):
            self.runtime.close()
        self.assertTrue(completion.query())
        self.assertFalse(self.runtime.block_manager.used_block_ids)

    def test_lightweight_promotion_event_covers_late_gpu_slot_metadata(self):
        from nanovllm.speculative.jetspec import prefill as module
        self.runtime.configure_optimizations(lightweight=True, feature_storage=True)
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        metadata_ready = torch.cuda.Event()
        original = module.canonical_slots
        context = self.runtime.begin_prefill([1, 2, 3])
        def delayed_slots(blocks, length, block_size, device):
            if context.remaining_tokens == 0:
                slots = original(blocks, length, block_size, device)
                # Delay after host->device page tensor construction, whose
                # blocking copy can otherwise consume an earlier test sleep.
                if hasattr(torch.cuda, "_sleep"):
                    torch.cuda._sleep(20_000_000)
                slots = slots.clone()
                metadata_ready.record(producer)
                return slots
            return original(blocks, length, block_size, device)
        with torch.cuda.stream(producer), patch.object(module, "canonical_slots", side_effect=delayed_slots):
            request = self.runtime.prefill_step(context, 3)
            if hasattr(torch.cuda, "_sleep"):
                self.assertFalse(metadata_ready.query(), "promotion should still have pending slot metadata")
        with torch.cuda.stream(consumer):
            request.state.assert_round_invariant(validate_device=False)
            slots = request.state.logical_slots.clone()
            consumer.synchronize()
        self.assertTrue(metadata_ready.query(), "consumer readiness did not fence promotion metadata")
        self.assertEqual(slots.tolist(), [0, 1, 2])


class ChunkedQwenModelTest(unittest.TestCase):
    def setUp(self):
        # get_rope caches a shared nn.Module. This suite explicitly moves a
        # tiny Target to CUDA; it must not leave its cached RoPE on CUDA for
        # unrelated CPU model tests with the same geometry.
        get_rope.cache_clear()
        self.addCleanup(get_rope.cache_clear)
        torch.manual_seed(53)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, max_position_embeddings=128, attention_bias=False)
        with patch("torch.distributed.get_world_size", return_value=1), patch(
                "torch.distributed.get_rank", return_value=0):
            self.target = Qwen3ForCausalLM(config)
        for parameter in self.target.parameters():
            torch.nn.init.uniform_(parameter, -.1, .1)
        self.ids = torch.tensor([1, 4, 3, 2, 5, 9, 7])

    @torch.inference_mode()
    def test_actual_layerwise_model_matches_full_prefill_kv_hidden_and_taps(self):
        expected_hidden, expected_kv, expected_taps = self.target.model.forward_dense(
            self.ids, torch.arange(len(self.ids)), None, None, (0, 1))
        for chunks in ([7], [1] * 7, [2, 3, 2], [4, 3]):
            pool = torch.full((2, 2, 3, 4, 2, 4), 999.)
            # Non-contiguous physical canonical pages, including a boundary.
            slots = torch.tensor([8, 9, 10, 11, 0, 1, 2])
            hiddens, taps, start = [], [], 0
            for count in chunks:
                h, t = self.target.model.forward_dense_chunk(self.ids[start:start+count],
                    torch.arange(start, start + count), pool, slots[:start],
                    slots[start:start+count], (0, 1))
                hiddens.append(h)
                taps.append(t)
                start += count
            torch.testing.assert_close(torch.cat(hiddens), expected_hidden, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(torch.cat(taps), expected_taps, atol=2e-6, rtol=2e-5)
            for layer, (k, v) in enumerate(expected_kv):
                torch.testing.assert_close(pool[0, layer, slots // 4, slots % 4], k, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(pool[1, layer, slots // 4, slots % 4], v, atol=2e-6, rtol=2e-5)
            self.assertTrue(bool((pool[:, :, 1] == 999.).all()))

    def test_offset_mask_not_upper_left_sdpa_causality(self):
        mask = offset_causal_mask(3, 2, "cpu")
        self.assertTrue(torch.equal(mask, torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)))
        self.assertIsNone(offset_causal_mask(0, 2, "cpu"))
        with self.assertRaises(ValueError):
            offset_causal_mask(-1, 2, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    @torch.inference_mode()
    def test_actual_qwen_chunk_promotion_on_different_cuda_streams(self):
        runtime = make_serving_runtime()
        runtime.target = self.target.cuda()
        runtime.target_layer_ids = (0, 1)
        runtime.block_size = 8
        runtime.kv_pool = torch.zeros((2, 2, 8, 8, 2, 4), device="cuda")
        runtime.block_manager = BlockManager(8, 8)
        runtime.arena = TreeScratchArena(runtime.kv_pool, runtime.block_manager, 8)
        ids = self.ids.cuda()
        hidden, kv, taps = runtime.target.model.forward_dense(ids, torch.arange(len(ids), device="cuda"), None, None, (0, 1))
        expected_anchor = int(runtime.target.lm_head(hidden[-1:]).argmax())
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        try:
            with torch.cuda.stream(producer):
                context = runtime.begin_prefill(self.ids.tolist(), max_new_tokens=4)
                self.assertIsNone(runtime.prefill_step(context, 3))
            with torch.cuda.stream(consumer):
                request = runtime.prefill_step(context, 4)
                request.state.assert_round_invariant(validate_device=True)
                self.assertEqual(request.output_ids, [expected_anchor])
                torch.testing.assert_close(request.state.target_hidden[0], taps, atol=3e-6, rtol=3e-5)
                rows = pool_rows(runtime, request.state.logical_slots)
                for layer, (keys, values) in enumerate(kv):
                    torch.testing.assert_close(rows[0, layer], keys, atol=3e-6, rtol=3e-5)
                    torch.testing.assert_close(rows[1, layer], values, atol=3e-6, rtol=3e-5)
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
