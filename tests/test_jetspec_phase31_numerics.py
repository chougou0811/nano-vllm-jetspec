"""CPU checks for diagnostic chronology and independent semantic guards.

Importing these helpers does not instantiate an engine or load model weights.
The production runtime and existing qualification tests remain untouched.
"""
from collections import OrderedDict
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))
from jetspec_phase31_numerics import compare_path, locate, row_trace, semantic_checks


def semantic_fixture():
    # Parent indices, not a triangular causal mask, define this branched tree.
    tree = {
        "num_nodes": 4,
        "parent_indices": torch.tensor([-1, 0, 0, 1]),
        "depth": torch.tensor([0, 1, 1, 2]),
        "ancestor": torch.tensor([
            [True, False, False, False],
            [True, True, False, False],
            [True, False, True, False],
            [True, True, False, True],
        ]),
    }
    request = {"cache_len": 7, "committed": torch.arange(8).reshape(1, 8)}
    return {"trees": [tree], "requests": [request]}


def trace(records, selected_rows, stage_inputs=None):
    inputs = {} if stage_inputs is None else stage_inputs
    return SimpleNamespace(
        records=OrderedDict(records), selected_rows=tuple(selected_rows),
        stage_inputs=inputs,
        operator_shapes={key: {"input": list(value.shape)} for key, value in inputs.items()},
    )


class Phase31NumericsCPU(unittest.TestCase):
    def test_independent_visibility_and_depth_accept_valid_branched_tree(self):
        self.assertTrue(semantic_checks(semantic_fixture()))

    def test_independent_visibility_rejects_causal_but_wrong_sibling(self):
        snapshot = semantic_fixture()
        # Node 2 is earlier in flat order, but is not node 3's ancestor.
        snapshot["trees"][0]["ancestor"][3, 2] = True
        with self.assertRaisesRegex(AssertionError, "ancestor visibility"):
            semantic_checks(snapshot)

    def test_independent_visibility_rejects_missing_ancestor(self):
        snapshot = semantic_fixture()
        snapshot["trees"][0]["ancestor"][3, 1] = False
        with self.assertRaisesRegex(AssertionError, "ancestor visibility"):
            semantic_checks(snapshot)

    def test_independent_depth_rejects_flat_index_used_as_depth(self):
        snapshot = semantic_fixture()
        snapshot["trees"][0]["depth"][3] = 3
        with self.assertRaisesRegex(AssertionError, "logical depth"):
            semantic_checks(snapshot)

    def test_uncached_anchor_invariant_rejects_length_mismatch(self):
        snapshot = semantic_fixture()
        snapshot["requests"][0]["cache_len"] = 8
        with self.assertRaisesRegex(AssertionError, "uncached-anchor"):
            semantic_checks(snapshot)

    def test_locate_uses_local_depth_not_global_row_or_flat_node_id(self):
        first = {
            "requests": [{"prompt_id": "chosen", "output_ids": [42]}],
            "trees": [{"depth": torch.tensor([0])}],
            "paths": {"chosen": [0]}, "query_offsets": [0, 1],
        }
        second = {
            "requests": [{"prompt_id": "other", "output_ids": [99]},
                         {"prompt_id": "chosen", "output_ids": [42, 43]}],
            "trees": [{"depth": torch.zeros(63, dtype=torch.long)},
                      {"depth": torch.tensor([0, 1, 1, 1, 1, 1, 1, 1])}],
            "paths": {"other": [0], "chosen": [0, 7]},
            "query_offsets": [0, 63, 71],
        }
        round_id, snapshot, request_id, offset = locate([first, second], "chosen", 3)
        self.assertEqual((round_id, request_id, offset), (1, 1, 1))
        self.assertIs(snapshot, second)
        self.assertEqual(second["query_offsets"][request_id] + second["paths"]["chosen"][offset], 70)
        # Logical prediction index is 3, while packed row is 70 and node ID is 7.
        self.assertEqual(locate([first, second], "chosen", 1)[0], 0)
        with self.assertRaisesRegex(AssertionError, "could not locate"):
            locate([first, second], "chosen", 70)

    def test_locate_returns_first_matching_snapshot(self):
        snapshot = {
            "requests": [{"prompt_id": "chosen", "output_ids": [42]}],
            "trees": [{"depth": torch.tensor([0])}],
            "paths": {"chosen": [0]},
        }
        self.assertEqual(locate([snapshot, copy.deepcopy(snapshot)], "chosen", 1)[0], 0)

    def test_row_trace_slices_every_stage_and_operator_input(self):
        selected = trace([
            ("layer_00.input_hidden", torch.tensor([[1., 2.], [3., 4.]])),
            ("layer_00.q_raw", torch.tensor([[5., 6.], [7., 8.]])),
        ], [63, 94], {"layer_00.q_raw": torch.tensor([[9., 10.], [11., 12.]])})
        row = row_trace(selected, 1)
        self.assertEqual(row.selected_rows, (94,))
        torch.testing.assert_close(row.records["layer_00.input_hidden"], torch.tensor([[3., 4.]]))
        torch.testing.assert_close(row.records["layer_00.q_raw"], torch.tensor([[7., 8.]]))
        torch.testing.assert_close(row.stage_inputs["layer_00.q_raw"], torch.tensor([[11., 12.]]))
        self.assertIs(row.operator_shapes, selected.operator_shapes)

    def test_compare_path_prioritizes_chronological_node_before_layer(self):
        packed = trace([
            ("layer_00.input_hidden", torch.zeros(2, 1)),
            ("layer_00.q_raw", torch.tensor([[0.], [1.]])),
            ("layer_20.attention_output", torch.tensor([[1.], [0.]])),
        ], [4, 8])
        serial = [trace([
            ("layer_00.input_hidden", torch.zeros(1, 1)),
            ("layer_00.q_raw", torch.zeros(1, 1)),
            ("layer_20.attention_output", torch.zeros(1, 1)),
        ], [0]) for _ in range(2)]
        comparison = compare_path(packed, serial)
        self.assertEqual(comparison["earliest_chronological_node"]["path_offset"], 0)
        self.assertEqual(comparison["earliest_chronological_node"]["stage"], "layer_20.attention_output")
        self.assertEqual(comparison["by_path_offset"][1]["earliest_divergence"]["stage"], "layer_00.q_raw")
        self.assertNotIn("layer_00.q_raw", comparison["by_path_offset"][0]["stages"])

    def test_compare_path_equal_rows_have_no_origin(self):
        packed = trace([("layer_00.input_hidden", torch.tensor([[1.], [2.]]))], [0, 1])
        serial = [trace([("layer_00.input_hidden", torch.tensor([[value]]))], [0])
                  for value in (1., 2.)]
        result = compare_path(packed, serial)
        self.assertIsNone(result["earliest_chronological_node"])
        self.assertTrue(all(row["all_compared_stages_bitwise_equal"] for row in result["by_path_offset"]))


if __name__ == "__main__":
    unittest.main()
