import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "benchmarks" / "jetspec_tree_profile_report.py"
spec = importlib.util.spec_from_file_location("tree_profile_report", path)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def event(category, name, ts, duration, *, args=None, pid=7, tid=7):
    return {"ph": "X", "cat": category, "name": name, "ts": ts, "dur": duration,
            "args": args or {}, "pid": pid, "tid": tid}


class TreeProfileReportTests(unittest.TestCase):
    def test_driver_launch_correlates_triton_with_target_without_external_id(self):
        trace = {"traceEvents": [
            event("user_annotation", "phase5.target.verify", 0, 1000),
            event("cuda_driver", "cuLaunchKernelEx", 10, 5, args={"correlation": 17}),
            event("kernel", "_packed_paged_tree_fp32", 500, 100,
                  args={"correlation": 17}, pid=0, tid=1),
        ]}
        result = driver.attribute_trace(trace)
        self.assertEqual(result["exclusive_stages"]["target_verify"]["gpu_kernel_ms"], .1)
        self.assertEqual(result["tree_attention_share_target_verify_cuda_kernels"], 1)
        self.assertEqual(result["launch_calls"], {"cuLaunchKernelEx": 1})

    def test_partition_with_prefill_does_not_use_verify_denominator_for_all(self):
        trace = {"traceEvents": [
            event("user_annotation", "phase5.target.verify", 0, 100),
            event("cuda_driver", "cuLaunchKernelEx", 10, 5, args={"correlation": 17}),
            event("kernel", "_packed_paged_tree_fp32", 500, 100,
                  args={"correlation": 17}, pid=0, tid=1),
            event("cuda_runtime", "cudaLaunchKernel", 200, 5, args={"correlation": 18}),
            event("kernel", "pytorch_flash::flash_fwd_kernel", 1000, 100,
                  args={"correlation": 18}, pid=0, tid=1),
        ]}
        result = driver.attribute_trace(trace)
        self.assertEqual(result["tree_attention_share_all_cuda_kernels"], .5)
        self.assertEqual(result["tree_attention_share_target_verify_cuda_kernels"], 1)
        self.assertTrue(result["prefill_observed"])
        self.assertEqual(result["launch_count"], 2)

    def test_cpu_external_id_shape_confirms_mlp(self):
        trace = {"traceEvents": [
            event("user_annotation", "phase5.target.verify", 0, 100),
            event("cpu_op", "aten::mm", 10, 5,
                  args={"External id": 17, "Input Dims": [[376, 4096], [4096, 12288]]}),
            event("kernel", "cutlass_gemm", 500, 200,
                  args={"External id": 17}, pid=0, tid=1),
        ]}
        result = driver.attribute_trace(trace)
        key = "target_verify:mlp_gemm_shape_confirmed"
        self.assertEqual(result["exclusive_stage_pieces"][key]["gpu_kernel_ms"], .2)
        self.assertEqual(result["gemm_input_shapes"][key][0]["input_dims"],
                         [[376, 4096], [4096, 12288]])

    def test_cuda_copies_not_counted_as_kernel_time(self):
        result = driver.attribute_trace({"traceEvents": [
            event("gpu_memcpy", "Memcpy DtoH", 0, 100),
            event("cuda_runtime", "cudaStreamSynchronize", 0, 100),
        ]})
        self.assertEqual(result["summed_cuda_kernel_ms"], 0)
        self.assertEqual(result["synchronization_count"], 1)
        self.assertIsNone(result["tree_attention_share_all_cuda_kernels"])


if __name__ == "__main__":
    unittest.main()
