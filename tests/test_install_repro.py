"""Portable install/preflight tests. No GPU, downloads or model weights needed."""
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


doctor = load_script("jetspec_environment_check", "tools/check_environment.py")
smoke = load_script("jetspec_portable_smoke", "examples/jetspec_serving.py")


class InstallReproTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.target, self.draft = self.base / "target", self.base / "draft"
        self.target.mkdir()
        self.draft.mkdir()
        self.target_config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                              "num_hidden_layers": 36, "hidden_size": 4096, "vocab_size": 151936}
        self.draft_config = {"model_type": "qwen3", "architectures": ["DFlashDraftModel"],
                             "hidden_size": 4096, "vocab_size": 151936, "block_size": 16,
                             "dflash_config": {"causal_head": True, "target_layer_ids": [1, 9, 17, 25, 33]}}
        self.write_configs()
        for path in (self.target, self.draft):
            (path / "model.safetensors").touch()

    def write_configs(self):
        (self.target / "config.json").write_text(json.dumps(self.target_config))
        (self.draft / "config.json").write_text(json.dumps(self.draft_config))

    def test_qualified_pair_config(self):
        result = doctor.checkpoint_pair(self.target, self.draft)
        self.assertEqual(result["target_layer_ids"], [1, 9, 17, 25, 33])

    def test_wrong_target_family(self):
        self.target_config["model_type"] = "llama"
        self.write_configs()
        with self.assertRaisesRegex(ValueError, "model_type"):
            doctor.checkpoint_pair(self.target, self.draft)

    def test_wrong_pair_dimensions(self):
        self.target_config["hidden_size"] = 1024
        self.write_configs()
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            doctor.checkpoint_pair(self.target, self.draft)

    def test_wrong_head_geometry(self):
        self.draft_config["block_size"] = 32
        self.write_configs()
        with self.assertRaisesRegex(ValueError, "block_size"):
            doctor.checkpoint_pair(self.target, self.draft)

    def test_wrong_tap_index(self):
        self.draft_config["dflash_config"]["target_layer_ids"] = [36]
        self.write_configs()
        with self.assertRaisesRegex(ValueError, "target_layer_ids"):
            doctor.checkpoint_pair(self.target, self.draft)

    def test_valid_but_unqualified_tap_order_rejected(self):
        self.draft_config["dflash_config"]["target_layer_ids"] = [9, 1, 17, 25, 33]
        self.write_configs()
        with self.assertRaisesRegex(ValueError, "qualified Draft target_layer_ids"):
            doctor.checkpoint_pair(self.target, self.draft)

    def test_missing_weights(self):
        (self.draft / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "no safetensors"):
            doctor.checkpoint_pair(self.target, self.draft)

    def test_partial_download_rejected(self):
        (self.target / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"model.layer": "model-00002-of-00002.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "missing weight shards"):
            doctor.checkpoint_pair(self.target, self.draft)

    def test_smoke_explicit_paths_and_chunk_defaults(self):
        args = smoke.arguments(["--target", str(self.target), "--draft", str(self.draft)])
        self.assertTrue(args.chunked_prefill)
        self.assertEqual(args.prefill_chunk_size, 256)
        self.assertEqual(args.max_tokens, 32)
        self.assertLess(args.gpu_memory_utilization, 0.8)

    def test_smoke_nonchunked_control(self):
        args = smoke.arguments(["--target", str(self.target), "--draft", str(self.draft),
                                "--no-chunked-prefill"])
        self.assertFalse(args.chunked_prefill)

    def test_smoke_rejects_invalid_budget(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            smoke.arguments(["--target", str(self.target), "--draft", str(self.draft),
                             "--prefill-chunk-size", "0"])

    def test_help_without_site_packages_or_gpu(self):
        for relative in ("tools/check_environment.py", "examples/jetspec_serving.py"):
            # -S removes site-packages, so even torch is unavailable.
            result = subprocess.run([sys.executable, "-S", str(ROOT / relative), "--help"],
                                    cwd=self.base, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)

    def test_smoke_delivery_and_cleanup_without_gpu(self):
        instances = []

        class FakeEngine:
            def __init__(self, *args, **kwargs):
                self.done = self.exited = self.disabled = False
                self.ids = []
                self.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2],
                                                 decode=lambda ids, **k: str(ids))
                instances.append(self)

            def configure_jetspec(self, *args, **kwargs):
                self.options = kwargs

            def add_request(self, prompt, params, *, request_id, tree_budget):
                self.ids.append(request_id)

            def is_finished(self):
                return self.done

            def step(self):
                self.last_step_info = {"blocked": False, "events": [
                    {"request_id": key, "kind": kind, "token_ids": [10, 11]}
                    for key in self.ids for kind in ("tokens", "finished")]}
                self.done = True

            def disable_jetspec(self):
                self.disabled = True

            def exit(self):
                self.exited = True

        args = smoke.arguments(["--target", str(self.target), "--draft", str(self.draft)])
        fake_modules = {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                                                __version__="fake"),
                        "nanovllm": SimpleNamespace(LLM=FakeEngine,
                                                    SamplingParams=lambda **kw: kw)}
        with patch.dict(sys.modules, fake_modules):
            result = smoke.run(args)
        self.assertTrue(result["exactly_once_delivery"])
        self.assertEqual([row["token_ids"] for row in result["requests"]], [[10, 11], [10, 11]])
        self.assertTrue(instances[0].options["enable_chunked_prefill"])
        self.assertTrue(instances[0].exited and instances[0].disabled)

    def test_package_metadata_pins(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest("TOML metadata parser available in Python >= 3.11")
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        self.assertIn("chougou0811/nano-vllm-jetspec", project["urls"]["Repository"])
        self.assertTrue(all(not dependency.startswith("flash-attn") for dependency in project["dependencies"]))
        self.assertIn(doctor.JETSPEC_REVISION, project["optional-dependencies"]["jetspec"][0])
        constraints = (ROOT / "requirements/constraints-cu128.txt").read_text()
        for name, version in doctor.QUALIFIED.items():
            self.assertIn(f"{name}=={version}", constraints)


if __name__ == "__main__":
    unittest.main()
