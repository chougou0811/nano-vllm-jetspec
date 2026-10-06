import unittest
from types import SimpleNamespace

import torch

from nanovllm.speculative.jetspec.target_graph import PackedTargetGraph, graph_signature


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


if __name__ == "__main__":
    unittest.main()
