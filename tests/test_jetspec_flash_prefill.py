"""Explicit backend policy and offset-causal FlashAttention prefill gates."""
import importlib.util
import sys
import unittest
from unittest.mock import patch

import torch

from nanovllm.speculative.jetspec.flash_prefill import (
    FlashPrefillMetadata, flash_causal_prefill, require_flash_attention,
)
from tests.test_jetspec_continuous import make_serving_runtime


class FlashPrefillMetadataTest(unittest.TestCase):
    def test_full_prefill_reuses_one_cumulative_length_tensor(self):
        meta = FlashPrefillMetadata.build(17, 17, "cpu")
        self.assertIs(meta.cu_query, meta.cu_key)
        self.assertEqual(meta.cu_query.tolist(), [0, 17])
        self.assertEqual(meta.cu_query.dtype, torch.int32)

    def test_chunk_has_suffix_query_and_longer_key_metadata(self):
        meta = FlashPrefillMetadata.build(3, 100, "cpu")
        self.assertEqual(meta.cu_query.tolist(), [0, 3])
        self.assertEqual(meta.cu_key.tolist(), [0, 100])
        self.assertEqual((meta.query_length, meta.key_length), (3, 100))

    def test_empty_or_shorter_key_is_rejected(self):
        for query, key in ((0, 1), (2, 1), (-1, 3)):
            with self.assertRaises(ValueError):
                FlashPrefillMetadata.build(query, key, "cpu")

    def test_metadata_mismatch_and_cpu_execution_fail_explicitly(self):
        q = torch.zeros(3, 4, 64, dtype=torch.bfloat16)
        kv = torch.zeros(4, 2, 64, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "geometry mismatch"):
            flash_causal_prefill(q, kv, kv, FlashPrefillMetadata.build(3, 3, "cpu"), 0.125)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            flash_causal_prefill(q, kv, kv, FlashPrefillMetadata.build(3, 4, "cpu"), 0.125)

    def test_missing_external_package_never_silently_falls_back(self):
        require_flash_attention.cache_clear()
        try:
            with patch.dict(sys.modules, {"flash_attn": None}):
                with self.assertRaisesRegex(RuntimeError, "no SDPA fallback"):
                    require_flash_attention()
        finally:
            require_flash_attention.cache_clear()

    def test_runtime_default_policy_and_invalid_reconfiguration_are_atomic(self):
        runtime = make_serving_runtime()
        try:
            runtime.configure_optimizations(lightweight=True, feature_storage=True)
            self.assertEqual(runtime._attention_backend, "sdpa")
            with self.assertRaisesRegex(ValueError, "backend"):
                runtime.configure_optimizations(attention_backend="not-a-backend")
            self.assertTrue(runtime._lightweight)
            self.assertTrue(runtime._feature_storage)
            with self.assertRaisesRegex(ValueError, "batched"):
                runtime.configure_optimizations(attention_backend="flash_attn")
            self.assertEqual(runtime._attention_backend, "sdpa")
            with patch("nanovllm.speculative.jetspec.flash_prefill.require_flash_attention"):
                runtime.configure_optimizations(batched_draft=True, attention_backend="flash_attn")
            self.assertEqual(runtime._attention_backend, "flash_attn")
            self.assertEqual(runtime._prefill_attention_backend, "sdpa")
        finally:
            runtime.close()


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("flash_attn"),
                     "real CUDA FlashAttention is not installed")
class FlashPrefillCUDATest(unittest.TestCase):
    def test_full_and_offset_chunk_match_independent_fp32_reference(self):
        torch.manual_seed(8)
        for prefix, query in ((0, 17), (97, 3), (1024, 31)):
            with self.subTest(prefix=prefix, query=query):
                q = torch.randn(query, 4, 64, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(prefix + query, 2, 64, device="cuda", dtype=torch.bfloat16)
                v = torch.randn_like(k)
                meta = FlashPrefillMetadata.build(query, prefix + query, "cuda")
                out = flash_causal_prefill(q, k, v, meta, 0.125)
                keys, values = k.float().repeat_interleave(2, 1), v.float().repeat_interleave(2, 1)
                scores = torch.einsum("qhd,khd->hqk", q.float(), keys) * 0.125
                allowed = torch.arange(prefix + query, device="cuda")[None, :] <= (
                    prefix + torch.arange(query, device="cuda")[:, None])
                scores.masked_fill_(~allowed[None], float("-inf"))
                expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values)
                self.assertTrue(bool(torch.isfinite(out).all()))
                torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
