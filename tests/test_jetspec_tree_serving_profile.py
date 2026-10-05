import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
spec = importlib.util.spec_from_file_location("tree_serving_profile", BENCHMARKS / "jetspec_tree_serving_profile.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class FakeProfile:
    def __init__(self):
        self.starts = self.stops = 0

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1


class TreeServingProfileTests(unittest.TestCase):
    def test_select_case_does_not_rebuild_manifest(self):
        cases = [{"concurrency": 8, "output_cap_scale": 512, "specs": [object()]}]
        self.assertIs(driver.select_case({"cases": cases}, 8, 512), cases[0])
        with self.assertRaises(AssertionError):
            driver.select_case({"cases": cases}, 4, 512)

    def test_select_case_rejects_duplicate(self):
        case = {"concurrency": 8, "output_cap_scale": 512}
        with self.assertRaises(AssertionError):
            driver.select_case({"cases": [case, case]}, 8, 512)

    def test_cpu_categories_are_explicit_proxies(self):
        rows = [{"operator": name, "calls": 1} for name in
                ("aten::mm", "aten::mul", "aten::item", "aten::remainder", "aten::cat", "unrelated")]
        result = driver.categorize_cpu(rows)
        self.assertEqual(result["gemm"], [rows[0]])
        self.assertEqual(result["norm_and_rope_elementwise_proxy"], [rows[1]])
        self.assertEqual(result["cpu_materialization"], [rows[2]])
        self.assertEqual(result["slot_mapping_and_scatter"], [rows[3]])
        self.assertEqual(result["tensor_copy_and_pack"], [rows[4]])

    def test_window_counts_only_verification_rounds_and_restores_step(self):
        profile = FakeProfile()
        fake_torch = SimpleNamespace(profiler=SimpleNamespace(
            profile=lambda **kwargs: profile,
            ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda")))
        infos = iter([
            {"events": [{"request_id": "r", "kind": "tokens", "token_ids": list(range(128))}]},
            {"events": [], "verification": None},
            {"events": [], "verification": {"node_counts": [31], "total_query_tokens": 31,
                "requests": [{"request_id": "r", "output_block": [1, 2], "accepted_draft_length": 1}]}},
            {"events": [], "verification": {"node_counts": [31], "total_query_tokens": 31,
                "requests": [{"request_id": "r", "output_block": [3], "accepted_draft_length": 0}]}},
            {"events": []},
        ])
        engine = SimpleNamespace(last_step_info=None)
        def original():
            engine.last_step_info = next(infos)
            return "production-return"
        engine.step = original
        with patch.dict(sys.modules, {"torch": fake_torch}):
            window = driver.LateWindow(engine, 128, 2, "unused.trace.json")
        with window:
            for _ in range(5):
                self.assertEqual(engine.step(), "production-return")
        self.assertIs(engine.step, original)
        self.assertEqual((profile.starts, profile.stops), (1, 1))
        self.assertEqual(len(window.rows), 2)
        self.assertEqual(window.rows[0]["node_counts"], [31])
        self.assertEqual(window.rows[0]["requests"][0]["emitted_tokens"], 2)

    def test_window_restores_on_exception(self):
        profile = FakeProfile()
        fake_torch = SimpleNamespace(profiler=SimpleNamespace(
            profile=lambda **kwargs: profile,
            ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda")))
        engine = SimpleNamespace(step=lambda: None)
        original = engine.step
        with patch.dict(sys.modules, {"torch": fake_torch}):
            window = driver.LateWindow(engine, 0, 4, "unused")
        with self.assertRaises(RuntimeError):
            with window:
                window.profile, window.active = profile, True
                profile.start()
                raise RuntimeError("test failure")
        self.assertIs(engine.step, original)
        self.assertEqual(profile.stops, 1)


if __name__ == "__main__":
    unittest.main()
