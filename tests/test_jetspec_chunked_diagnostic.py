"""CPU-only tests for portable chunked numerical qualification controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
from jetspec_chunked_numeric_diagnostic import chronological_forward, independent_mask, fp32_parameters
from jetspec_chunked_prefill import close_target_capture, numeric_diagnostic_evidence, tensor_metrics


class ChunkedNumericControlsTests(unittest.TestCase):
    def test_bitwise_gate_distinguishes_signed_zero(self):
        positive = torch.tensor([0.], device="cpu")
        negative = torch.tensor([-0.], device="cpu")
        self.assertFalse(tensor_metrics(positive, negative, 0)["bitwise_equal"])

    def test_numerically_equal_different_dtype_rejected(self):
        with self.assertRaisesRegex(AssertionError, "dtype"):
            tensor_metrics(torch.tensor([1.], device="cpu", dtype=torch.float32),
                           torch.tensor([1.], device="cpu", dtype=torch.bfloat16), 0)

    def test_independent_offset_mask(self):
        from nanovllm.speculative.jetspec.prefill import offset_causal_mask
        for prefix in (0, 1, 7, 255, 256):
            for count in (1, 7, 256):
                actual = independent_mask(prefix, count, "cpu")
                reference = offset_causal_mask(prefix, count, "cpu")
                if prefix:
                    self.assertTrue(torch.equal(actual, reference))
                else:
                    self.assertIsNone(actual)
                    self.assertIsNone(reference)

    def test_fp32_parameter_cast_restores_exceptional_backing(self):
        layer = torch.nn.Linear(7, 3, dtype=torch.bfloat16, device="cpu")
        originals = [(parameter, parameter.data_ptr(), parameter.detach().clone()) for parameter in layer.parameters()]
        with self.assertRaisesRegex(ValueError, "injected"):
            with fp32_parameters(layer):
                self.assertTrue(all(p.dtype == torch.float32 for p in layer.parameters()))
                raise ValueError("injected")
        for parameter, pointer, value in originals:
            self.assertEqual(parameter.dtype, torch.bfloat16)
            self.assertEqual(parameter.data_ptr(), pointer)
            self.assertTrue(torch.equal(parameter, value))

    def test_chronological_reference_past_and_ragged_geometry(self):
        calls = []
        class Model:
            layers = (None, None)
            def forward_dense(self, ids, positions, previous, mask, taps):
                calls.append((positions.clone(), previous, mask))
                value = torch.stack((ids.float(), positions.float()), dim=-1)
                keys = [(value.unsqueeze(1) + i, value.unsqueeze(1) + 100 + i) for i in range(2)]
                return value, keys, value
        target = type("Target", (), {"model": Model()})()
        ids = torch.tensor([10, 11, 12, 13, 14, 15, 16], device="cpu")
        hidden, kv, taps = chronological_forward(target, ids, [1, 3, 1, 2], (0, 1))
        self.assertEqual(tuple(kv.shape), (2, 2, 7, 1, 2))
        self.assertTrue(torch.equal(hidden, taps))
        self.assertIsNone(calls[0][1])
        for cursor, call in zip((1, 4, 5), calls[1:]):
            self.assertEqual(call[1][0][0].shape[0], cursor)
            self.assertTrue(torch.equal(call[1][0][0][:, 0, 0], ids[:cursor].float()))
            self.assertTrue(torch.equal(call[2], independent_mask(cursor, len(call[0]), "cpu")))

    def test_target_capture_removed_before_draft_reuses_head(self):
        head = torch.nn.Identity()
        captured = []
        hook = head.register_forward_hook(lambda _m, _a, out: captured.append(out[-1].detach().cpu()))
        head(torch.tensor([[3., 2.]], device="cpu"))
        target = close_target_capture(captured, hook)
        head(torch.tensor([[0., 5.]], device="cpu"))
        self.assertEqual(len(captured), 1)
        self.assertEqual(int(target.argmax()), 0)

    def test_missing_target_capture_rejected(self):
        class Hook:
            def remove(self):
                raise AssertionError("must not silently close a nonexistent prediction")
        with self.assertRaisesRegex(AssertionError, "did not produce"):
            close_target_capture([], Hook())


class RequiredNumericEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "diagnostic.json"
        self.source = {"production_sha256": "production", "script_sha256": "harness"}
        self.models = {"target": {"path": "/explicit/target"}, "draft": {"path": "/explicit/draft"}}
        self.environment = {"torch": "version", "cuda": "version", "gpu": "device"}
        metric = {"passed": True, "bound": 2e-4}
        self.evidence = {"passed": True, "allocator_clean": True, "source_unchanged": True,
            "source": self.source, "models": self.models, "environment": self.environment,
            "diagnostic_sha256": hashlib.sha256((BENCHMARKS / "jetspec_chunked_numeric_diagnostic.py").read_bytes()).hexdigest(),
            "trace_helper": {"path": str(BENCHMARKS / "jetspec_layer_trace.py"),
                "sha256": hashlib.sha256((BENCHMARKS / "jetspec_layer_trace.py").read_bytes()).hexdigest()},
            "arguments": {"length": 33, "chunk": 1},
            "same_shape_independent_reference": {"logical_prompt_equal": True,
                "kv": {"bitwise_equal": True}, "taps": {"bitwise_equal": True}},
            "bf16_layer_major_independent_alignment": dict.fromkeys(("full_kv", "chunk_kv", "full_taps", "chunk_taps"), True),
            "fp32_control": {"matmul_allow_tf32": False, "sdpa_backend": "MATH", "roundoff_bound": 2e-4},
            "fp32_layer_major": {"layers": [dict(layer=i, key=metric, value=metric, output_hidden=metric) for i in range(2)],
                "final_hidden": metric, "taps": metric, "target_logits": metric}}

    def check(self, evidence=None):
        self.path.write_text(json.dumps(self.evidence if evidence is None else evidence))
        return numeric_diagnostic_evidence(self.path, self.source, self.models, self.environment, 2)

    def test_complete_evidence_passes_and_references_actual_sha(self):
        result = self.check()
        self.assertTrue(result["passed"])
        self.assertEqual(result["layers"], 2)
        self.assertEqual(result["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_source_checkpoint_environment_and_helper_must_match(self):
        for field, altered in (("source", {"production_sha256": "wrong", "script_sha256": "harness"}),
                               ("models", {}), ("environment", dict(self.environment, gpu="other")),
                               ("diagnostic_sha256", "wrong"), ("trace_helper", {"sha256": "wrong"})):
            with self.subTest(field=field), self.assertRaises((AssertionError, KeyError)):
                self.check(dict(self.evidence, **{field: altered}))

    def test_partial_failed_or_reordered_layers_rejected(self):
        for layers in ([], list(reversed(self.evidence["fp32_layer_major"]["layers"])),
                       [dict(layer=0, key={"passed": False, "bound": 2e-4}, value={"passed": True, "bound": 2e-4},
                             output_hidden={"passed": True, "bound": 2e-4}), self.evidence["fp32_layer_major"]["layers"][1]]):
            evidence = deepcopy(self.evidence)
            evidence["fp32_layer_major"]["layers"] = layers
            with self.assertRaises(AssertionError):
                self.check(evidence)

    def test_not_allowed_to_relax_fp32_envelope(self):
        evidence = deepcopy(self.evidence)
        evidence["fp32_control"]["roundoff_bound"] = .1
        with self.assertRaises(AssertionError):
            self.check(evidence)

    def test_shape_control_cleanup_and_tf32_are_hard_gates(self):
        for mutate in (lambda e: e.update(allocator_clean=False),
                       lambda e: e["same_shape_independent_reference"]["kv"].update(bitwise_equal=False),
                       lambda e: e["fp32_control"].update(matmul_allow_tf32=True)):
            evidence = deepcopy(self.evidence)
            mutate(evidence)
            with self.assertRaises(AssertionError):
                self.check(evidence)


if __name__ == "__main__":
    unittest.main()
