"""CPU oracle/metadata checks and explicitly enabled CUDA tree-kernel gates.

RUN_JETSPEC_TREE_KERNEL_GPU_TESTS=1 python -m unittest discover -s tests \
    -p test_jetspec_tree_kernel.py -v

Normal test discovery never starts GPU work or loads model weights.
"""
from pathlib import Path
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

import jetspec_tree_kernel_qualification as qualification
from jetspec_numeric_oracles import FP32_ATTENTION_BOUND, fp64_attention, numerical_metrics, parent_chain


class TreeKernelMetadataCPU(unittest.TestCase):
    def test_frozen_numerical_gates_are_not_relaxed(self):
        self.assertEqual(FP32_ATTENTION_BOUND, 1e-4)
        self.assertEqual(qualification.BF16_BOUND, 2 ** -6)
        self.assertFalse(numerical_metrics(torch.ones(8) + .0002, torch.ones(8))["within_fixed_bound"])
        self.assertTrue(numerical_metrics(torch.ones(8) + .00001, torch.ones(8))["within_fixed_bound"])

    def test_random_topological_tree_matches_independent_parent_chains(self):
        for nodes in (1, 2, 31, 47, 63):
            parents = qualification.parents_for(nodes)
            mask = qualification.mask_from_parents(parents)
            self.assertEqual(parents[0], -1)
            for node in range(nodes):
                self.assertEqual(mask[node].nonzero().flatten().tolist(), parent_chain(parents, node))
                self.assertTrue(mask[node, node])
                self.assertFalse(bool(mask[node, node + 1:].any()))

    def test_all_boundary_fixtures_have_independent_ragged_addressing(self):
        for prefix in qualification.PREFIX_LENGTHS:
            with self.subTest(prefix=prefix):
                fixture = qualification.synthetic_fixture((prefix, 0, max(0, prefix - 1)),
                                                          head_dim=8, kv_heads=1)
                meta = fixture["metadata"]
                self.assertEqual(meta.query_offsets, (0, 63, 94, 141))
                self.assertEqual(meta.prefix_lengths, (prefix, 0, max(0, prefix - 1)))
                live = torch.cat(fixture["prefix_slots"] + fixture["node_slots"])
                self.assertEqual(live.numel(), live.unique().numel())
                flat_k = fixture["k"].view(-1, 1, 8)
                self.assertTrue(bool(torch.isfinite(flat_k[live]).all()))
                inactive = torch.ones(flat_k.shape[0], dtype=torch.bool)
                inactive[live] = False
                self.assertTrue(bool(torch.isnan(flat_k[inactive]).all()))
                # Deliberately not contiguous logical or physical tree storage.
                for slots in fixture["node_slots"]:
                    self.assertFalse(bool(torch.all(slots[1:] - slots[:-1] == 1)))

    def test_fp64_oracle_does_not_consult_production_bias(self):
        fixture = qualification.synthetic_fixture((1, 0, 2), counts=(3, 2, 3),
                                                   head_dim=8, kv_heads=1, dtype=torch.float32)
        meta = fixture["metadata"]
        output = torch.empty_like(fixture["q"])
        for request, count in enumerate(meta.node_counts_host):
            for node in range(count):
                slots = torch.cat((fixture["prefix_slots"][request],
                    fixture["node_slots"][request][parent_chain(fixture["parents"][request], node)]))
                output[meta.query_offsets[request] + node] = fp64_attention(
                    fixture["q"][meta.query_offsets[request] + node],
                    fixture["k"][slots // 256, slots % 256], fixture["v"][slots // 256, slots % 256],
                    fixture["scale"], fixture["groups"]).float()
        meta.qq_bias.fill_(float("nan"))
        checks = qualification.oracle_rows(fixture, output, all_rows=True)
        self.assertEqual(len(checks), 8)
        self.assertTrue(all(check["within_fixed_bound"] for check in checks))

    def test_fp64_oracle_rejects_even_finite_large_numeric_error(self):
        fixture = qualification.synthetic_fixture((0,), counts=(1,), head_dim=8, kv_heads=1,
                                                   dtype=torch.float32)
        with self.assertRaisesRegex(AssertionError, "FP32 parent-chain oracle failed"):
            qualification.oracle_rows(fixture, torch.ones_like(fixture["q"]) * 100)

    def test_nonfinite_valid_operand_cannot_pass_oracle(self):
        fixture = qualification.synthetic_fixture((1,), counts=(1,), head_dim=8, kv_heads=1,
                                                   dtype=torch.float32)
        slot = fixture["prefix_slots"][0][0]
        fixture["k"][slot // 256, slot % 256] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite valid operand"):
            qualification.oracle_rows(fixture, torch.zeros_like(fixture["q"]))

    def test_bf16_quantization_is_not_scored_as_pre_round_fp32(self):
        reference = torch.full((16,), 1.003, dtype=torch.float64)
        diagnostic = numerical_metrics(reference.to(torch.bfloat16), reference)
        self.assertFalse(diagnostic["fixed_bound_applicable"])
        self.assertTrue(diagnostic["bf16_quantization"]["actual_output_is_bf16"])
        self.assertEqual(diagnostic["bf16_quantization"]["nearest_bf16_unequal_elements"], 0)

    def test_byte_comparison_handles_nan_and_dtype(self):
        a = torch.tensor([float("nan"), 1.0])
        self.assertTrue(qualification.bits_equal(a, a.clone()))
        self.assertFalse(qualification.bits_equal(a, torch.tensor([float("nan"), 2.0])))
        self.assertFalse(qualification.bits_equal(a, a.double()))

    def test_empty_fixture_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prefix/count"):
            qualification.synthetic_fixture((1,), counts=(31, 47), head_dim=8)
        with self.assertRaisesRegex(ValueError, "head geometry"):
            qualification.synthetic_fixture((1,), counts=(31,), head_dim=8, groups=0)

    def test_lifecycle_gate_requires_real_eos_not_only_cap_completion(self):
        rows = [dict(request_id=name, status=status, token_ids=tokens, max_tokens=64)
                for name, status, tokens in (("eos", "finished", [2, 99]),
                    ("running-cancel", "cancelled", [3]), ("queued-cancel", "cancelled", []),
                    ("one-token", "finished", [4]), ("late", "finished", [5]), ("refill", "finished", [6]))]
        result = dict(requests=rows, dynamic_live_arrival_seen=True)
        self.assertTrue(all(qualification.lifecycle_semantics(result, {99}).values()))
        rows[0]["token_ids"][-1] = 1
        with self.assertRaisesRegex(AssertionError, "EOS"):
            qualification.lifecycle_semantics(result, {99})

    def test_bad_kernel_options_are_rejected_before_any_cuda_work(self):
        from nanovllm.speculative.jetspec.tree_attention import packed_tree_attention_gqa
        fixture = qualification.synthetic_fixture((1,), counts=(1,), head_dim=8, kv_heads=1)
        args = [fixture[name] for name in ("q", "k", "v", "metadata", "scale", "groups")]
        for tile in (0, 3, True):
            with self.assertRaisesRegex(ValueError, "query_tile"):
                packed_tree_attention_gqa(*args, query_tile=tile)
        for warps in (0, 3, True):
            with self.assertRaisesRegex(ValueError, "num_warps"):
                packed_tree_attention_gqa(*args, num_warps=warps)

    def test_candidate_layout_options_are_filtered_by_explicit_signature(self):
        def prefix(*args, output_dtype=None, num_warps=4):
            return args, output_dtype, num_warps
        def grouped(*args, output_dtype=None, query_tile=1, num_warps=4):
            return args, output_dtype, query_tile, num_warps
        self.assertEqual(qualification.candidate_options(prefix, query_tile=4, num_warps=8),
                         dict(num_warps=8))
        self.assertEqual(qualification.candidate_options(grouped, query_tile=4, num_warps=8),
                         dict(query_tile=4, num_warps=8))
        operands = (object(), object(), object())
        with patch.object(qualification, "candidate_entry", return_value=(None, prefix)):
            received, dtype, warps = qualification.launch_candidate(operands, "fake:prefix",
                output_dtype=torch.float32, query_tile=4, num_warps=8)
        self.assertTrue(all(a is b for a, b in zip(operands, received)))
        self.assertEqual(dtype, torch.float32)
        self.assertEqual(warps, 8)

    def test_missing_candidate_cannot_fall_back_to_reference(self):
        with self.assertRaisesRegex(ValueError, "module:function"):
            qualification.candidate_entry("automatic")
        with self.assertRaises(AttributeError):
            qualification.candidate_entry("tree_attention:missing_function")

    def test_candidate_source_and_effective_options_are_recorded(self):
        info = qualification.candidate_identity("tree_attention:packed_tree_attention_gqa",
                                                  query_tile=2, num_warps=4)
        self.assertEqual(info["executed_options"], dict(query_tile=2, num_warps=4))
        self.assertEqual(len(info["module_sha256"]), 64)
        self.assertEqual(len(info["function_source_sha256"]), 64)
        self.assertIn("output_dtype", info["supported_parameters"])

    def test_failed_trained_envelope_is_recorded_before_assertion(self):
        passing = qualification.frozen.tensor_metrics(torch.ones(8), torch.ones(8),
                                                       bound=qualification.BF16_BOUND)
        failed = qualification.frozen.tensor_metrics(torch.ones(8) + .02, torch.ones(8),
                                                      bound=qualification.BF16_BOUND)
        check = {"attention_operator_controls": [{"layer": 0}], "all_layer_tree_kv": failed,
            "final_hidden": passing, "target_taps": passing, "lm_head": passing,
            "tree_kv_by_layer": [{"layer": 0, "metrics": failed}],
            "argmax_flip_witness": {"all_flip_witnesses_consistent": True},
            "canonical_history_byte_exact": True, "candidate_repeat_bitwise_exact": {"logits": True}}
        records = []
        with self.assertRaisesRegex(AssertionError, "full-network fixed BF16"):
            qualification.record_and_require_trained_check(check, observed_layers=1,
                expected_layers=1, expected_controls=1, record_check=records.append)
        self.assertEqual(records, [check])
        self.assertFalse(records[0]["passed"])
        self.assertFalse(records[0]["gates"]["full_network_fixed_bf16_envelope"])
        self.assertEqual(records[0]["tree_kv_by_layer"][0]["metrics"], failed)
        self.assertGreater(records[0]["all_layer_tree_kv"]["relative_rms_error"], 2 ** -6)

    def test_passing_trained_check_is_recorded_exactly_once(self):
        passing = qualification.frozen.tensor_metrics(torch.ones(8), torch.ones(8),
                                                       bound=qualification.BF16_BOUND)
        check = {"attention_operator_controls": [{"layer": 0}],
            **{key: passing for key in ("all_layer_tree_kv", "final_hidden", "target_taps", "lm_head")},
            "tree_kv_by_layer": [{"layer": 0, "metrics": passing}],
            "argmax_flip_witness": {"all_flip_witnesses_consistent": True},
            "canonical_history_byte_exact": True, "candidate_repeat_bitwise_exact": {"logits": True}}
        records = []
        qualification.record_and_require_trained_check(check, observed_layers=1, expected_layers=1,
            expected_controls=1, record_check=records.append)
        self.assertEqual(len(records), 1)
        self.assertTrue(check["passed"])
        self.assertTrue(all(check["gates"].values()))

    def test_fixed_q408_controls_explicitly_disable_new_chunked_default(self):
        calls = []
        engine = SimpleNamespace(is_finished=lambda: True,
            scheduler=SimpleNamespace(max_num_seqs=8),
            _jetspec_scheduler=SimpleNamespace(max_num_seqs=8),
            configure_jetspec=lambda draft, **kwargs: calls.append((draft, kwargs)))
        args = SimpleNamespace(draft="draft-model", max_model_len=4096)
        original = qualification.serving.set_mode
        with qualification.qualification_serving_modes(engine, args):
            qualification.serving.set_mode(engine, "jetspec", args.draft, 8)
            self.assertEqual(calls[-1][0], args.draft)
            policy = calls[-1][1]
            self.assertFalse(policy["enable_chunked_prefill"])
            self.assertEqual(policy["max_prefill_tokens"], 4096)
            self.assertEqual(policy["attention_backend"], "sdpa")
            self.assertEqual(policy["optimization"], "serving")
            self.assertEqual(engine._jetspec_scheduler.max_num_seqs, 8)
            with self.assertRaisesRegex(AssertionError, "unexpected"):
                qualification.serving.set_mode(engine, "ordinary", args.draft, 8)
        self.assertIs(qualification.serving.set_mode, original)

    def test_qualification_serving_mode_cannot_reconfigure_live_engine(self):
        engine = SimpleNamespace(is_finished=lambda: False)
        args = SimpleNamespace(draft="draft-model", max_model_len=4096)
        with qualification.qualification_serving_modes(engine, args):
            with self.assertRaisesRegex(AssertionError, "idle"):
                qualification.serving.set_mode(engine, "jetspec", args.draft, 8)


GPU_ENABLED = os.getenv("RUN_JETSPEC_TREE_KERNEL_GPU_TESTS") == "1"


@unittest.skipUnless(GPU_ENABLED, "set RUN_JETSPEC_TREE_KERNEL_GPU_TESTS=1 for CUDA qualification")
class TreeKernelCUDA(unittest.TestCase):
    def test_tile64_page256_boundaries_and_mixed_ragged_requests(self):
        for dtype in (torch.bfloat16, torch.float32):
            for prefix in qualification.PREFIX_LENGTHS:
                fixture = qualification.synthetic_fixture((prefix, max(0, prefix - 1), prefix // 2),
                                                           device="cuda", dtype=dtype)
                for tile in (1, 2):
                    with self.subTest(dtype=dtype, prefix=prefix, query_tile=tile):
                        check = qualification.synthetic_case(fixture, query_tile=tile, all_rows=prefix <= 65)
                        self.assertTrue(check["fp32_output_round_trip_bitwise_exact"])
                        self.assertTrue(check["same_shape_repeat_bitwise_exact"])

    def test_request_and_offbranch_isolation_for_all_grouped_query_tiles(self):
        fixture = qualification.synthetic_fixture((65, 0, 257), device="cuda")
        for tile in (1, 2, 4):
            with self.subTest(tile=tile):
                result = qualification.isolation_case(fixture, query_tile=tile)
                self.assertTrue(all(result["other_request_neighbors_bitwise_exact"]))
                self.assertTrue(result["off_branch_ancestor_path_bitwise_exact"])
                self.assertTrue(result["perturbed_request_changed"])

    def test_single_request_and_packed_neighbor_shapes_do_not_change_chosen_output(self):
        from nanovllm.speculative.jetspec.packed_metadata import PackedTreeMetadata
        from nanovllm.speculative.jetspec.tree_attention import packed_tree_attention_gqa
        fixture = qualification.synthetic_fixture((65, 0, 257), device="cuda")
        meta = fixture["metadata"]
        local_meta = PackedTreeMetadata.build([65], [meta.block_tables_host[0]],
            fixture["node_slots"][:1], [qualification.mask_from_parents(fixture["parents"][0]).cuda()], 256)
        for tile in (1, 2, 4):
            packed = packed_tree_attention_gqa(fixture["q"], fixture["k"], fixture["v"], meta,
                                              fixture["scale"], fixture["groups"], query_tile=tile)
            local = packed_tree_attention_gqa(fixture["q"][:63], fixture["k"], fixture["v"], local_meta,
                                             fixture["scale"], fixture["groups"], query_tile=tile)
            self.assertTrue(qualification.bits_equal(packed[:63], local))

    def test_strided_qkv_and_gqa_geometry_are_supported(self):
        for groups, head_dim in ((1, 64), (4, 128), (8, 128)):
            fixture = qualification.synthetic_fixture((63, 64, 65), device="cuda", groups=groups,
                                                       head_dim=head_dim)
            # Two-times leading/inter-head strides, contiguous final feature.
            q = torch.empty((fixture["q"].shape[0] * 2, fixture["q"].shape[1] * 2, head_dim),
                            dtype=fixture["q"].dtype, device="cuda")[::2, ::2]
            q.copy_(fixture["q"])
            fixture["q"] = q
            for name in ("k", "v"):
                value = fixture[name]
                strided = torch.empty((value.shape[0] * 2, value.shape[1], value.shape[2] * 2, head_dim),
                                      dtype=value.dtype, device="cuda")[::2, :, ::2]
                strided.copy_(value)
                fixture[name] = strided
            qualification.synthetic_case(fixture, query_tile=1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
