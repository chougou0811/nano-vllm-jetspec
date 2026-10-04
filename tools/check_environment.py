#!/usr/bin/env python3
"""Check the qualified JetSpec installation without loading model weights.

No CUDA calls are made unless --require-cuda is supplied. Checkpoint validation
is local-only and does not download or execute checkpoint-provided Python code.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


JETSPEC_REVISION = "2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f"
QUALIFIED = {"torch": "2.8.0+cu128", "triton": "3.4.0",
             "transformers": "4.57.6", "huggingface-hub": "0.36.2"}


def checkpoint_pair(target, draft):
    """Validate this fork's supported trained pair without allocating weights."""
    configs = []
    for path in (Path(target), Path(draft)):
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "qwen3":
            raise ValueError(f"unsupported model_type at {path}")
        shards = list(path.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"no safetensors weights at {path}")
        index = path / "model.safetensors.index.json"
        if index.is_file():
            expected = set(json.loads(index.read_text())["weight_map"].values())
            missing = [name for name in expected if not (path / name).is_file()]
            if missing:
                raise ValueError(f"missing weight shards at {path}: {missing}")
        configs.append(config)
    target_cfg, draft_cfg = configs
    for name in ("hidden_size", "vocab_size"):
        if target_cfg.get(name) != draft_cfg.get(name):
            raise ValueError(f"Target/Draft {name} mismatch")
    if target_cfg.get("architectures") != ["Qwen3ForCausalLM"]:
        raise ValueError("Target must be the Qwen3 causal LM, not the draft checkpoint")
    if draft_cfg.get("architectures") != ["DFlashDraftModel"]:
        raise ValueError("Draft must be the trained DFlashDraftModel checkpoint")
    draft_options = draft_cfg.get("dflash_config", {})
    layers = draft_options.get("target_layer_ids", [])
    if not layers or any(not 0 <= layer < target_cfg["num_hidden_layers"] for layer in layers):
        raise ValueError("invalid Draft target_layer_ids")
    if layers != [1, 9, 17, 25, 33]:
        raise ValueError("qualified Draft target_layer_ids must be [1, 9, 17, 25, 33]")
    if target_cfg["num_hidden_layers"] != 36 or target_cfg["hidden_size"] != 4096:
        raise ValueError("qualified Target must be the 36-layer, hidden-size 4096 Qwen3-8B")
    if draft_cfg.get("block_size") != 16 or not draft_options.get("causal_head"):
        raise ValueError("the qualified head requires block_size=16 and causal_head=True")
    return {"target_layers": target_cfg["num_hidden_layers"], "target_layer_ids": layers,
            "block_size": draft_cfg["block_size"],
            "note": "config/shard presence checks only; not weight hashing or numerical qualification"}


def official_revision(module):
    direct = metadata.distribution("jetspec").read_text("direct_url.json")
    if direct:
        info = json.loads(direct)
        commit = info.get("vcs_info", {}).get("commit_id")
        if commit:
            return commit
    # Developers may legitimately use an editable clone of the same pinned code.
    result = subprocess.run(["git", "-C", str(Path(module.__file__).parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def diagnose(*, target=None, draft=None, require_cuda=False, strict_pins=False):
    checks = []
    def record(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    record("python", "pass" if (3, 11) <= sys.version_info[:2] < (3, 13) else "fail",
           platform.python_version())
    record("platform", "pass" if sys.platform == "linux" else "warn",
           f"{sys.platform}; qualified on Linux / Python 3.12")
    for name, expected in QUALIFIED.items():
        try:
            found = metadata.version(name)
            status = "pass" if found == expected else ("fail" if strict_pins else "warn")
            record(name, status, {"installed": found, "qualified": expected})
        except metadata.PackageNotFoundError:
            record(name, "fail", "not installed")
    modules = {}
    for name in ("torch", "triton", "transformers", "jetspec", "nanovllm"):
        try:
            module = importlib.import_module(name)
            modules[name] = module
            record(f"import:{name}", "pass", str(Path(module.__file__).resolve()))
        except Exception as exc:
            record(f"import:{name}", "fail", f"{type(exc).__name__}: {exc}")
    if "nanovllm" in modules:
        record("fork_serving_api", "pass" if hasattr(modules["nanovllm"].LLM, "configure_jetspec") else "fail",
               "LLM.configure_jetspec must exist; installing upstream nano-vLLM is not sufficient")
    if "jetspec" in modules:
        try:
            revision = official_revision(modules["jetspec"])
            record("official_jetspec_revision", "pass" if revision == JETSPEC_REVISION else
                   ("fail" if strict_pins else "warn"),
                   {"installed": revision, "qualified": JETSPEC_REVISION})
            from jetspec import DraftHeadTreeDrafter, load_draft_head  # noqa: F401
            from jetspec.models.draft_head import DFlashDraftModel  # noqa: F401
            record("trained_draft_api", "pass", "official model and adapter import successfully")
        except Exception as exc:
            record("trained_draft_api", "fail", f"{type(exc).__name__}: {exc}")
    try:
        flash = metadata.version("flash-attn")
        record("flash_attention", "warn", f"installed {flash}; optional and outside the qualified stack")
    except metadata.PackageNotFoundError:
        record("flash_attention", "pass", "absent (expected); ordinary attention requires enforce_eager=True with the slower SDPA fallback")
    if target is not None:
        try:
            record("checkpoint_pair", "pass", checkpoint_pair(target, draft))
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(str(target), local_files_only=True)
            if config.dtype != modules["torch"].bfloat16:
                raise ValueError("qualified Target must load as BF16; check transformers/config compatibility")
            record("hf_config_dtype", "pass", str(config.dtype))
        except Exception as exc:
            record("checkpoint_pair_or_hf_config", "fail", f"{type(exc).__name__}: {exc}")
    if require_cuda:
        try:
            torch = modules["torch"]
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; check the CUDA wheel, driver, and GPU visibility")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("this JetSpec path requires BF16 support")
            record("cuda", "pass", {"device": torch.cuda.get_device_name(0),
                                     "torch_cuda": torch.version.cuda,
                                     "total_bytes": torch.cuda.get_device_properties(0).total_memory})
        except Exception as exc:
            record("cuda", "fail", f"{type(exc).__name__}: {exc}")
    return {"ok": not any(row["status"] == "fail" for row in checks), "checks": checks,
            "scope": "installation preflight, not a GPU correctness/performance qualification"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--strict-pins", action="store_true")
    args = parser.parse_args(argv)
    if (args.target is None) != (args.draft is None):
        parser.error("--target and --draft must be supplied together")
    result = diagnose(**vars(args))
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
