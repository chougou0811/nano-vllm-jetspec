"""CPU-only parsing/aggregation tests; importing the driver does not run CUDA."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import jetspec_flash_draft_profile as profile


class FlashDraftProfileTests(unittest.TestCase):
    def test_microtimings_keep_ten_raw_runs_and_median_not_fastest(self):
        rows = [{"phase": "timed", "mode": mode, "repeat": repeat,
                 "cpu_wall_ms": repeat + 1, "cuda_event_ms": 2 * repeat + 1}
                for mode in profile.MODES for repeat in range(10)]
        summary = profile.summarize_microtimings(rows)
        self.assertEqual(summary["sdpa"]["median_cpu_wall_ms_including_tail_sync"], 5.5)
        self.assertEqual(summary["flash_attn"]["median_cuda_event_timeline_ms"], 10)
        self.assertEqual(summary["sdpa"]["raw_cpu_wall_ms"], list(range(1, 11)))
        self.assertIn("no serving speedup", summary["interpretation"])
        with self.assertRaisesRegex(AssertionError, "exactly ten"):
            profile.summarize_microtimings(rows[:-1])
        rows[-1]["repeat"] = 0
        with self.assertRaisesRegex(AssertionError, "duplicate"):
            profile.summarize_microtimings(rows)

    def test_trace_counts_real_kernels_launch_apis_and_sync_separately(self):
        trace = {"traceEvents": [
            {"ph": "X", "cat": "kernel", "name": "flash_forward", "dur": 20},
            {"ph": "X", "cat": "kernel", "name": "flash_forward", "dur": 30},
            {"ph": "X", "cat": "kernel", "name": "cat_copy", "dur": 10},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "dur": 2},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernelExC", "dur": 3},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "dur": 7},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaDeviceSynchronize", "dur": 8},
            {"ph": "X", "cat": "cpu_op", "name": "aten::_scaled_dot_product_flash_attention", "dur": 500},
            {"ph": "M", "cat": "kernel", "name": "metadata", "dur": 1000}]}
        result = profile.summarize_trace(trace)
        self.assertEqual(result["cuda_kernel_count"], 3)
        self.assertAlmostEqual(result["summed_cuda_kernel_duration_ms"], .06)
        self.assertEqual(result["cuda_launch_call_count"], 2)
        self.assertEqual(result["cuda_synchronization_call_count"], 2)
        self.assertEqual(result["kernel_names"][0],
            {"name": "flash_forward", "calls": 2, "total_gpu_ms": .05})

    def test_attention_dispatch_selection_keeps_actual_pytorch_backend_names(self):
        rows = [{"name": name} for name in ("aten::scaled_dot_product_attention",
            "aten::_scaled_dot_product_flash_attention", "aten::_scaled_dot_product_efficient_attention",
            "FlashAttnVarlenFunc", "aten::mm", "aten::cat")]
        self.assertEqual(profile.attention_operators(rows), rows[:4])

    def test_original_function_evidence_removes_profiler_and_never_replaces_api(self):
        def original():
            return 11
        with profile.ProposalCallEvidence(original) as evidence:
            self.assertEqual(original(), 11)
            original()
        self.assertIsNone(sys.getprofile())
        result = evidence.result("flash_attn")
        self.assertEqual(result["original_flash_attn_varlen_calls"], 2)
        with profile.ProposalCallEvidence(original) as evidence:
            pass
        self.assertEqual(evidence.result("sdpa")["original_flash_attn_varlen_calls"], 0)
        with self.assertRaisesRegex(AssertionError, "disagree"):
            evidence.result("flash_attn")

    def test_operator_time_parser_does_not_confuse_inclusive_with_self(self):
        event = SimpleNamespace(key="aten::mm", count=8,
            self_cpu_time_total=3000, cpu_time_total=4000,
            self_device_time_total=5000, device_time_total=6000)
        parsed = profile.operator_rows(SimpleNamespace(key_averages=lambda: [event]))
        self.assertEqual(parsed, [{"name": "aten::mm", "calls": 8,
            "self_cpu_ms": 3, "inclusive_cpu_ms": 4,
            "self_gpu_ms": 5, "inclusive_gpu_ms": 6}])


if __name__ == "__main__":
    unittest.main()
