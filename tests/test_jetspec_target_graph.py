import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch

from nanovllm.speculative.jetspec.target_graph import PackedTargetGraph, graph_signature
from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata


class TargetGraphContractTests(unittest.TestCase):
    def metadata(self, prefix=(128, 1024), nodes=(63, 31)):
        return SimpleNamespace(total_queries=sum(nodes), prefix_lengths=prefix,
            qq_bias=torch.empty(sum(n * n for n in nodes)))

    def test_signature_does_not_cache_prefix_values(self):
        self.assertEqual(graph_signature(self.metadata(), prefix_backend=True),
                         graph_signature(self.metadata((1023, 2048)), prefix_backend=True))

    def test_signature_preserves_exact_gemm_rows_and_mask_capacity(self):
        self.assertNotEqual(graph_signature(self.metadata(), prefix_backend=True),
                            graph_signature(self.metadata(nodes=(63, 47)), prefix_backend=True))

    def test_signature_separates_reference_dispatch(self):
        meta = self.metadata()
        self.assertNotEqual(graph_signature(meta, prefix_backend=True),
                            graph_signature(meta, prefix_backend=False))

    def test_cpu_is_not_a_graph_fallback(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            PackedTargetGraph(None, torch.empty((2, 1, 1, 256, 8, 128)), (), 4096)


class FakeStream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, stream):
        self.log.append((self.name, "wait_stream", stream.name))

    def wait_event(self, event):
        self.log.append((self.name, "wait_event"))

    def synchronize(self):
        self.log.append((self.name, "synchronize"))


class FakeEvent:
    def __init__(self, log):
        self.log = log

    def record(self, stream):
        self.log.append(("event", "record", stream.name))

    def synchronize(self):
        self.log.append(("event", "synchronize"))


class FakeGraph:
    def __init__(self, log):
        self.log = log

    def pool(self):
        return "shared-graph-pool"

    def replay(self):
        # This is ONLY an orchestration mock, not numerical CUDA-graph evidence.
        self.log.append(("graph", "replay"))


class TargetGraphOrchestrationCPU(unittest.TestCase):
    """Exercise real staging/capture logic while every CUDA API is a mock."""

    def setUp(self):
        self.log = []
        self.current = FakeStream("current", self.log)
        self.capture = FakeStream("capture", self.log)
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        # Tensor storage and all arithmetic remain on CPU. Override only the
        # CUDA eligibility property, never create a CUDA tensor or real stream.
        self.patches.enter_context(patch.object(torch.Tensor, "is_cuda",
                                               new_callable=PropertyMock, return_value=True))
        self.patches.enter_context(patch("torch.cuda.Stream", return_value=self.capture))
        self.patches.enter_context(patch("torch.cuda.current_stream", return_value=self.current))
        self.patches.enter_context(patch("torch.cuda.Event", side_effect=lambda: FakeEvent(self.log)))
        self.patches.enter_context(patch("torch.cuda.stream", side_effect=lambda stream: nullcontext()))
        self.patches.enter_context(patch("torch.cuda.synchronize",
                                        side_effect=lambda device: self.log.append(("device", "synchronize"))))
        self.patches.enter_context(patch("torch.cuda.CUDAGraph", side_effect=lambda: FakeGraph(self.log)))
        self.capture_context = self.patches.enter_context(patch("torch.cuda.graph",
                                                                side_effect=lambda *args, **kwargs: nullcontext()))
        self.prefix_dispatch = self.patches.enter_context(patch(
            "nanovllm.speculative.jetspec.paged_backend._use_prefix_tree_attention", return_value=False))
        self.kv_pool = torch.zeros((2, 1, 4, 256, 2, 8), dtype=torch.bfloat16)
        self.forward = Mock(side_effect=lambda tokens, *args: (tokens.float().unsqueeze(1),
                                                             tokens.float().unsqueeze(1) + 100))
        self.target = SimpleNamespace(model=SimpleNamespace(forward_packed_tree=self.forward,
            layers=[SimpleNamespace(self_attn=SimpleNamespace(num_heads=8, num_kv_heads=2, head_dim=8))]),
            lm_head=lambda hidden: hidden)

    def metadata(self, *, prefixes=(3, 5), nodes=(2, 3), slots_start=512):
        counts = tuple(nodes)
        flat = torch.arange(slots_start, slots_start + sum(counts))
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        slots = [flat[offsets[i]:offsets[i + 1]] for i in range(len(counts))]
        table_cursor = 0
        tables = []
        for length in prefixes:
            pages = (length + 255) // 256
            tables.append(list(range(table_cursor, table_cursor + pages)))
            table_cursor += pages
        return PackedTreeMetadata.build(prefixes, tables, slots,
            [torch.ones(n, n, dtype=torch.bool).tril() for n in counts], 256,
            request_ids=[f"request-{i}" for i in range(len(counts))])

    def inputs(self, meta):
        return torch.arange(meta.total_queries), torch.arange(meta.total_queries) + 3

    def make_graph(self, *, max_graphs=16, max_model_len=4096):
        return PackedTargetGraph(self.target, self.kv_pool, (0,), max_model_len,
                                 max_graphs=max_graphs)

    def test_warmup_and_capture_write_only_real_staged_inputs(self):
        graph, meta = self.make_graph(), self.metadata()
        tokens, positions = self.inputs(meta)
        graph.verify(tokens, positions, meta)
        self.assertEqual(self.forward.call_count, 3, "two warmups and one capture")
        self.assertEqual(graph.captures, 1)
        self.assertEqual(graph.replays, 1)
        entry = next(iter(graph.entries.values()))
        for call in self.forward.call_args_list:
            self.assertTrue(torch.equal(call.args[0], tokens))
            self.assertTrue(torch.equal(call.args[1], positions))
            self.assertTrue(torch.equal(call.args[3].tree_slots, meta.tree_slots))
        self.assertEqual(entry["metadata"].block_tables.shape, (2, 16))
        self.assertTrue(bool((entry["metadata"].block_tables[:, 1:] == -1).all()))
        self.assertIn(("current", "wait_stream", "capture"), self.log)
        self.assertIn(("event", "record", "current"), self.log)

    def test_graph_hit_refreshes_swapped_ragged_counts_prefix_slots_and_tokens(self):
        graph, first = self.make_graph(), self.metadata()
        graph.verify(*self.inputs(first), first)
        # Same exact rows and sum(N_i^2), different per-request ownership/layout.
        second = self.metadata(prefixes=(7, 9), nodes=(3, 2), slots_start=769)
        tokens, positions = (value + 37 for value in self.inputs(second))
        graph.verify(tokens, positions, second)
        self.assertEqual(graph.captures, 1)
        self.assertEqual(graph.replays, 2)
        self.assertEqual(self.forward.call_count, 3, "graph hit cannot rerun Python forward")
        entry = next(iter(graph.entries.values()))
        self.assertTrue(torch.equal(entry["tokens"], tokens))
        self.assertTrue(torch.equal(entry["positions"], positions))
        for field in ("query_to_request", "query_local_row", "prefix_lens", "tree_slots",
                      "qq_bias", "qq_bias_offsets", "node_counts", "cu_seqlens_q"):
            self.assertTrue(torch.equal(getattr(entry["metadata"], field), getattr(second, field)), field)
        self.assertIn(("current", "wait_event"), self.log)

    def test_warmup_runtime_or_keyboard_exception_fences_capture_before_return(self):
        for failure in (RuntimeError("warmup failure"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                graph, meta = self.make_graph(), self.metadata()
                self.log.clear()
                with patch.object(graph, "_forward", side_effect=failure):
                    with self.assertRaises(type(failure)):
                        graph.verify(*self.inputs(meta), meta)
                self.assertEqual(graph.entries, {})
                self.assertEqual(graph.captures, 0)
                fence = self.log.index(("capture", "synchronize"))
                handoff = self.log.index(("current", "wait_stream", "capture"))
                self.assertLess(fence, handoff, "exception must fence before transaction rollback")

    def test_capture_body_failure_also_fences_and_never_caches_failed_graph(self):
        graph, meta = self.make_graph(), self.metadata()
        outputs = (torch.zeros(5, 1), torch.zeros(5, 1), torch.zeros(5, 1))
        with patch.object(graph, "_forward", side_effect=[outputs, outputs, RuntimeError("capture failure")]):
            with self.assertRaisesRegex(RuntimeError, "capture failure"):
                graph.verify(*self.inputs(meta), meta)
        self.assertIn(("capture", "synchronize"), self.log)
        self.assertEqual(graph.entries, {})
        self.assertIsNone(graph.pool)
        self.assertEqual(graph.replays, 0)

    def test_bounded_cache_falls_back_eager_without_allocating_or_evicting(self):
        graph, first = self.make_graph(max_graphs=1), self.metadata()
        graph.verify(*self.inputs(first), first)
        original_entry = next(iter(graph.entries.values()))
        second = self.metadata(nodes=(2, 2))
        with patch.object(graph, "_new_entry", side_effect=AssertionError("cache must remain bounded")):
            logits, taps = graph.verify(*self.inputs(second), second)
        self.assertEqual(logits.shape[0], second.total_queries)
        self.assertEqual(taps.shape[0], second.total_queries)
        self.assertEqual(len(graph.entries), 1)
        self.assertIs(next(iter(graph.entries.values())), original_entry)
        self.assertEqual(graph.eager_fallbacks, 1)
        self.assertEqual(graph.captures, 1)

    def test_pool_identity_or_invalid_address_rejected_before_any_forward(self):
        graph, meta = self.make_graph(), self.metadata()
        graph.kv_pool = self.kv_pool.clone()
        with self.assertRaisesRegex(RuntimeError, "live KV pool changed"):
            graph.verify(*self.inputs(meta), meta)
        self.forward.assert_not_called()
        graph = self.make_graph()
        invalid = replace(meta, tree_slot_ids=(1024, *meta.tree_slot_ids[1:]))
        with self.assertRaisesRegex(ValueError, "outside the KV pool"):
            graph.verify(*self.inputs(invalid), invalid)
        self.forward.assert_not_called()
        self.assertEqual(graph.entries, {})

    def test_prefix_table_capacity_rejected_before_staging(self):
        graph = self.make_graph(max_model_len=256)
        meta = self.metadata(prefixes=(257, 5), slots_start=768)
        with self.assertRaisesRegex(ValueError, "graph table capacity"):
            graph.verify(*self.inputs(meta), meta)
        self.forward.assert_not_called()

    def test_float_or_boolean_tokens_positions_are_rejected_before_capture(self):
        for field in ("tokens", "positions"):
            for dtype in (torch.float32, torch.bool):
                graph, meta = self.make_graph(), self.metadata()
                tokens, positions = self.inputs(meta)
                if field == "tokens":
                    tokens = tokens.to(dtype)
                else:
                    positions = positions.to(dtype)
                with self.subTest(field=field, dtype=dtype), self.assertRaisesRegex(ValueError, "integer"):
                    graph.verify(tokens, positions, meta)
                self.assertEqual(graph.entries, {})
        self.forward.assert_not_called()

    def test_prefix_and_reference_dispatch_use_separate_graphs(self):
        graph, meta = self.make_graph(), self.metadata()
        self.prefix_dispatch.side_effect = [False, True]
        graph.verify(*self.inputs(meta), meta)
        graph.verify(*self.inputs(meta), meta)
        self.assertEqual(graph.captures, 2)
        self.assertEqual(len(graph.entries), 2)

    def test_close_waits_for_replay_before_releasing_graph_entries(self):
        graph, meta = self.make_graph(), self.metadata()
        graph.verify(*self.inputs(meta), meta)
        graph.close()
        self.assertIn(("event", "synchronize"), self.log)
        self.assertEqual(graph.entries, {})
        self.assertIsNone(graph.pool)
        self.assertIsNone(graph.ready)


class TargetGraphPolicyCPU(unittest.TestCase):
    def runtime(self):
        from nanovllm.speculative.jetspec.batch_runtime import JetSpecBatchRuntime
        runtime = JetSpecBatchRuntime.__new__(JetSpecBatchRuntime)
        runtime._closed, runtime._active_transaction = False, None
        runtime.requests, runtime.prefills = {}, {}
        runtime._target_kernels, runtime._target_execution = "fused_rope", "cuda_graph"
        runtime._target_graph = Mock()
        runtime.target = SimpleNamespace(model=SimpleNamespace(layers=[
            SimpleNamespace(self_attn=SimpleNamespace()), SimpleNamespace(self_attn=SimpleNamespace())]))
        return runtime

    def test_kernel_policy_change_closes_graph_and_resets_all_layer_flags(self):
        runtime = self.runtime()
        old_graph = runtime._target_graph
        runtime.configure_optimizations(target_kernels="reference")
        old_graph.close.assert_called_once_with()
        self.assertIsNone(runtime._target_graph)
        for layer in runtime.target.model.layers:
            self.assertFalse(layer.self_attn._jetspec_tree_fusion)

    def test_unchanged_kernels_retain_warm_graph_across_execution_toggle(self):
        runtime = self.runtime()
        graph = runtime._target_graph
        runtime.configure_optimizations(target_execution="eager", target_kernels=None)
        self.assertIs(runtime._target_graph, graph)
        graph.close.assert_not_called()
        runtime.configure_optimizations(target_execution="cuda_graph", target_kernels="fused_rope")
        self.assertIs(runtime._target_graph, graph)
        self.assertTrue(all(layer.self_attn._jetspec_tree_fusion for layer in runtime.target.model.layers))

    def test_invalid_or_live_policy_change_cannot_release_graph(self):
        for kwargs in ({"target_execution": "unknown"}, {"target_kernels": "unknown"},
                       {"target_kernels": "fused"}, {"target_kernels": "fused_gemm"}):
            runtime = self.runtime()
            graph = runtime._target_graph
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                runtime.configure_optimizations(**kwargs)
            graph.close.assert_not_called()
        runtime = self.runtime()
        runtime.requests["live"] = object()
        graph = runtime._target_graph
        with self.assertRaisesRegex(RuntimeError, "live requests"):
            runtime.configure_optimizations(target_kernels="reference")
        graph.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
