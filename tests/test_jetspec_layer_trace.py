"""CPU regression for diagnostic instrumentation; no model weights/GPU required."""
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import Qwen3Config

from benchmarks.jetspec_layer_trace import LayerTrace, compare_traces, tensor_metrics
from nanovllm.models import qwen3
from nanovllm.speculative.jetspec import paged_backend
from nanovllm.utils.context import reset_context


class JetSpecLayerTraceTest(TestCase):
    def setUp(self):
        torch.manual_seed(31)
        config = Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=4, max_position_embeddings=128,
            attention_bias=False,
        )
        with patch("torch.distributed.get_world_size", return_value=1), patch(
            "torch.distributed.get_rank", return_value=0
        ):
            self.target = qwen3.Qwen3ForCausalLM(config)
        for parameter in self.target.parameters():
            torch.nn.init.uniform_(parameter, -0.1, 0.1)
        self.ids = torch.tensor([1, 2, 3])
        self.positions = torch.arange(3)
        reset_context()

    def tearDown(self):
        reset_context()

    def run_trace(self, ids=None, positions=None):
        with torch.inference_mode(), LayerTrace(
            self.target, [2], capture_full_layers=(0,)
        ) as trace:
            hidden, _, _ = self.target.model.forward_dense(
                self.ids if ids is None else ids,
                self.positions if positions is None else positions,
                None, None,
            )
            self.target.lm_head(hidden)
        return trace

    def test_actual_forward_repeat_shapes_and_patch_restore(self):
        original_linear = F.linear
        original_norm = qwen3._reference_rms_norm
        original_packed = paged_backend.packed_tree_attention
        left, right = self.run_trace(), self.run_trace()
        result = compare_traces(left, right)
        self.assertTrue(result["all_compared_stages_bitwise_equal"])
        self.assertTrue(result["trace_schema_equal"])
        self.assertIsNone(result["earliest_divergence"])
        self.assertEqual(len(left.records), 42)
        self.assertEqual(left.operator_shapes["layer_00.q_raw"]["input"], [3, 16])
        self.assertEqual(left.operator_shapes["layer_00.q_raw"]["weight"], [16, 16])
        self.assertEqual(left.full_records["layer_00.q_raw_input"].shape, (3, 16))
        self.assertEqual(left.records["layer_00.q_raw"].shape, (1, 16))
        self.assertEqual(left.records["layer_00.mlp_activation"].shape, (1, 24))
        self.assertEqual(left.records["lm_head"].shape, (1, 32))
        self.assertIs(F.linear, original_linear)
        self.assertIs(qwen3._reference_rms_norm, original_norm)
        self.assertIs(paged_backend.packed_tree_attention, original_packed)
        for layer in self.target.model.layers:
            self.assertNotIn("forward_dense", layer.__dict__)
            self.assertNotIn("forward_packed_tree", layer.__dict__)

    def test_exception_restores_all_patches_and_inherited_methods(self):
        original = (F.linear, F.silu, F.scaled_dot_product_attention,
                    qwen3._reference_rms_norm, paged_backend.packed_tree_attention)
        with self.assertRaisesRegex(ValueError, "selected trace rows"):
            with LayerTrace(self.target, [99]):
                self.target.model.forward_dense(self.ids, self.positions, None, None)
        self.assertEqual(original, (F.linear, F.silu, F.scaled_dot_product_attention,
                                   qwen3._reference_rms_norm, paged_backend.packed_tree_attention))
        for layer in self.target.model.layers:
            self.assertNotIn("forward_dense", layer.__dict__)
            self.assertNotIn("forward_paged_tree", layer.__dict__)
        # A failed trace must not poison the next ordinary Target execution.
        self.assertIn("lm_head", self.run_trace().records)

    def test_default_device_mode_reentrant_silu_is_observed_once(self):
        # Same TorchFunctionMode used by ModelRunner.set_default_device('cuda'),
        # but this regression requires no GPU and restores the mode on exit.
        original_silu = F.silu
        with torch.device("cpu"):
            left, right = self.run_trace(), self.run_trace()
        self.assertEqual(len(left.records), 42)
        self.assertTrue(compare_traces(left, right)["all_compared_stages_bitwise_equal"])
        self.assertIs(F.silu, original_silu)

    def test_earliest_divergence_reports_equal_projection_operands(self):
        left, right = self.run_trace(), self.run_trace()
        right.records["layer_00.q_raw"] = right.records["layer_00.q_raw"].clone()
        right.records["layer_00.q_raw"][0, 0] += 0.25
        result = compare_traces(left, right)
        first = result["earliest_divergence"]
        self.assertEqual(first["stage"], "layer_00.q_raw")
        self.assertEqual(first["unequal_elements"], 1)
        self.assertTrue(first["operator_input"]["bitwise_equal"])
        self.assertEqual(first["operator_shape_a"], first["operator_shape_b"])
        self.assertTrue(result["stages"]["layer_00.input_hidden"]["bitwise_equal"])

    def test_different_execution_shapes_are_recorded_without_reimplementing_forward(self):
        three = self.run_trace()
        four = self.run_trace(torch.tensor([1, 2, 3, 4]), torch.arange(4))
        result = compare_traces(three, four)
        self.assertTrue(result["stages"]["layer_00.input_hidden"]["bitwise_equal"])
        qkv = result["stages"]["layer_00.q_raw"]
        self.assertEqual(qkv["operator_shape_a"]["input"], [3, 16])
        self.assertEqual(qkv["operator_shape_b"]["input"], [4, 16])
        self.assertTrue(qkv["operator_input"]["bitwise_equal"])

    def test_metrics_use_bits_and_handle_scalar_signed_zero_and_nonfinite(self):
        zeros = tensor_metrics(torch.tensor(0.0), torch.tensor(-0.0))
        self.assertFalse(zeros["bitwise_equal"])
        self.assertEqual(zeros["unequal_elements"], 1)
        self.assertEqual(zeros["max_abs"], 0.0)
        values = torch.tensor([float("nan"), 1.0], dtype=torch.bfloat16)
        same = tensor_metrics(values, values.clone())
        self.assertTrue(same["bitwise_equal"])
        self.assertEqual(same["nonfinite_a"], 1)
        self.assertTrue(tensor_metrics(torch.zeros(2), torch.zeros(3))["incompatible"])

    def test_missing_additional_and_reordered_stages_cannot_pass_equality_gate(self):
        left = self.run_trace()
        for mutation in ("missing", "additional", "reordered"):
            with self.subTest(mutation=mutation):
                right = self.run_trace()
                if mutation == "missing":
                    right.records.pop("layer_00.q_norm")
                elif mutation == "additional":
                    right.records["unexpected_observation"] = right.records["lm_head"]
                else:
                    right.records.move_to_end("layer_00.input_hidden")
                result = compare_traces(left, right)
                self.assertFalse(result["trace_schema_equal"])
                self.assertFalse(result["all_compared_stages_bitwise_equal"])
                # A schema error is not misreported as a numeric divergence.
                self.assertIsNone(result["earliest_divergence"])
                if mutation == "missing":
                    self.assertEqual(result["missing_reference_stages"], ["layer_00.q_norm"])
                elif mutation == "additional":
                    self.assertEqual(result["reference_only_stages"], ["unexpected_observation"])

    def test_missing_operator_input_or_shape_cannot_pass_equality_gate(self):
        left = self.run_trace()
        for field in ("stage_inputs", "operator_shapes"):
            with self.subTest(field=field):
                right = self.run_trace()
                getattr(right, field).pop("layer_00.q_raw")
                result = compare_traces(left, right)
                self.assertFalse(result["trace_schema_equal"])
                self.assertFalse(result["all_compared_stages_bitwise_equal"])
                self.assertIsNone(result["earliest_divergence"])
