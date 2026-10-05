#!/usr/bin/env python3
"""Pristine upstream FlashAttention / frozen JetSpec system comparison.

Run each worker in its own environment/process, then --mode combine. This
driver never rewrites upstream/final production code, nor implements sampling.
The immutable token manifest is read from the earlier matched benchmark.
"""
from __future__ import annotations

import argparse
import atexit
from collections import Counter
from contextlib import AbstractContextManager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import jetspec_final_matched as previous

require, save, sha = previous.require, previous.save, previous.sha
UPSTREAM_HEAD = "df99418f7d6ca676550f4372cdc6e1521ce8c33d"
FINAL_HEAD = "b388330d45bf42adf7e0112319c031587d9828cb"
FINAL_PRODUCTION_SHA256 = "038fb0d746882451f915c3d028778198e508d488b0c06a8eba44354d726fa57c"
MODES = ("upstream_flashattn", "jetspec")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    require(len(manifest["cases"]) == 6, "expected original six-case manifest")
    require(sha(manifest["cases"]) == manifest["manifest_sha256"], "manifest content/hash mismatch")
    expected = [(c, o) for c in (1, 4, 8) for o in (128, 512)]
    for index, (case, dimensions) in enumerate(zip(manifest["cases"], expected)):
        require(case["case_index"] == index and
                (case["concurrency"], case["output_cap_scale"]) == dimensions, "manifest case matrix changed")
        require(sha(case["specs"]) == case["workload_sha256"], "workload/hash mismatch")
        require(len(case["specs"]) == 2 * dimensions[0], "original two-wave workload changed")
    return manifest


def source_identity(nano, official_jetspec=None):
    modules = [("nanovllm", nano)]
    if official_jetspec is not None:
        modules.append(("jetspec", official_jetspec))
    hashes = {}
    for label, module in modules:
        root = Path(module.__file__).resolve().parent
        for path in sorted(root.rglob("*.py")):
            hashes[label + ":" + str(path.relative_to(root))] = file_sha(path)
    return {"nano_source": str(Path(nano.__file__).resolve()),
            "production_sha256": sha(hashes), "source_file_sha256": hashes,
            "git": previous._helpers.git_provenance(nano),
            "official_jetspec_git": previous._helpers.git_provenance(official_jetspec)
            if official_jetspec is not None else None,
            "driver_sha256": file_sha(__file__),
            "metric_helper_sha256": file_sha(previous.__file__),
            "workload_helper_sha256": file_sha(previous.HELPER_PATH)}


class NativeDeliveryObserver:
    """Read original Sequence state after step; never intercept the scheduler."""
    def __init__(self):
        self.sequences = {}
        self.observed = {}
        self.finished = set()

    def add(self, public, sequence):
        require(public not in self.sequences, "duplicate public sequence")
        require(all(sequence.seq_id != seq.seq_id for seq in self.sequences.values()), "duplicate native sequence ID")
        self.sequences[public] = sequence
        self.observed[public] = sequence.num_prompt_tokens

    def events(self, terminal_outputs):
        terminal = dict(terminal_outputs)
        require(len(terminal) == len(terminal_outputs), "duplicate native terminal")
        events = []
        newly_finished = set()
        for public, seq in self.sequences.items():
            old = self.observed[public]
            require(seq.num_tokens >= old, "native completion history shrank")
            if seq.num_tokens != old:
                require(public not in self.finished, "native delivery after terminal")
                tokens = list(seq.token_ids[old:seq.num_tokens])
                require(len(tokens) == seq.num_tokens - old, "native delta length mismatch")
                events.append({"request_id": public, "kind": "tokens", "token_ids": tokens})
                self.observed[public] = seq.num_tokens
            if seq.is_finished and public not in self.finished:
                history = list(seq.token_ids[seq.num_prompt_tokens:seq.num_tokens])
                require(seq.seq_id in terminal and terminal[seq.seq_id] == history,
                        "native terminal outputs disagree with read-only history")
                events.append({"request_id": public, "kind": "finished", "token_ids": history})
                self.finished.add(public)
                newly_finished.add(seq.seq_id)
        require(newly_finished == set(terminal), "unobserved or repeated native terminal")
        return events


class FlashCallEvidence(AbstractContextManager):
    """Untimed warmup-only Python call tracing; no function replacement."""
    def __init__(self, attention):
        self.functions = {name: getattr(attention, name) for name in
                          ("flash_attn_varlen_func", "flash_attn_with_kvcache")}
        require(all(callable(f) and hasattr(f, "__code__") for f in self.functions.values()),
                "FlashAttention original Python wrappers are unavailable")
        self.codes = {f.__code__: name for name, f in self.functions.items()}
        self.counts = Counter()
        self.sdpa_calls = 0
        self.old_profile = None

    def _profile(self, frame, event, arg):
        if event == "call" and frame.f_code in self.codes:
            self.counts[self.codes[frame.f_code]] += 1
        elif event == "c_call" and getattr(arg, "__name__", "") == "scaled_dot_product_attention":
            self.sdpa_calls += 1

    def __enter__(self):
        self.old_profile = sys.getprofile()
        require(self.old_profile is None, "an existing profiler would confound Flash path qualification")
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(self.old_profile)

    def result(self):
        require(all(self.counts[name] > 0 for name in self.functions), "native warmup missed a Flash API")
        require(self.sdpa_calls == 0, "native warmup unexpectedly called SDPA")
        return {"phase": "untimed original-manifest warmup",
                "method": "sys.setprofile counts original imported function code objects; no monkeypatch",
                "counts": dict(self.counts), "sdpa_calls": self.sdpa_calls,
                "functions": {name: {"module": func.__module__,
                    "file": func.__code__.co_filename,
                    "file_sha256": file_sha(func.__code__.co_filename)}
                    for name, func in self.functions.items()},
                "profiler_removed_for_timed_samples": sys.getprofile() is None}


def upstream_idle(engine):
    manager = engine.scheduler.block_manager
    require(engine.is_finished() and not engine.scheduler.waiting and not engine.scheduler.running,
            "native scheduler did not fully drain")
    require(not manager.used_block_ids and len(manager.free_block_ids) == len(manager.blocks),
            "native KV allocations leaked")
    require(all(b.ref_count == 0 for b in manager.blocks), "native KV refcounts leaked")
    return {"used_blocks": 0, "scratch_blocks": 0, "free_blocks": len(manager.free_block_ids),
            "hash_entries": len(manager.hash_to_block_id), "all_non_scratch_pages_free": True}


def prepare_upstream(engine, concurrency):
    from nanovllm.engine.block_manager import BlockManager
    upstream_idle(engine)
    old = engine.scheduler.block_manager
    engine.scheduler.block_manager = BlockManager(len(old.blocks), old.block_size)
    engine.scheduler.max_num_seqs = concurrency
    engine.model_runner.config.max_num_seqs = concurrency
    result = upstream_idle(engine)
    require(result["hash_entries"] == 0, "native allocator was not reset cold")
    return result


def serve_upstream(engine, case, deadline):
    import torch
    from nanovllm import SamplingParams
    ledger = previous.Ledger(case["specs"])
    observer = NativeDeliveryObserver()
    submitted = set()
    mapping = {spec["request_id"]: spec["request_id"] for spec in case["specs"]}
    steps = peak_running = peak_waiting = 0
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident_allocated, resident_reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    with previous.AllocationObserver(engine.scheduler.block_manager) as allocation:
        start = time.perf_counter()
        while len(submitted) != len(case["specs"]) or not engine.is_finished():
            now = time.perf_counter() - start
            require(now < deadline, "native sample exceeded deadline")
            for spec in case["specs"]:
                public = spec["request_id"]
                if public not in submitted and now >= spec["arrival_s"]:
                    before = len(engine.scheduler.waiting)
                    returned = engine.add_request(spec["prompt"], SamplingParams(
                        temperature=0, max_tokens=spec["max_tokens"], ignore_eos=True))
                    require(returned is None and len(engine.scheduler.waiting) == before + 1,
                            "unexpected native add_request API")
                    observer.add(public, engine.scheduler.waiting[-1])
                    ledger.submit(public, time.perf_counter() - start)
                    submitted.add(public)
            peak_running = max(peak_running, len(engine.scheduler.running))
            peak_waiting = max(peak_waiting, len(engine.scheduler.waiting))
            if not engine.is_finished():
                terminals, _ = engine.step()
                delivered = time.perf_counter() - start
                ledger.consume(observer.events(terminals), delivered, mapping)
                peak_running = max(peak_running, len(engine.scheduler.running))
                peak_waiting = max(peak_waiting, len(engine.scheduler.waiting))
                steps += 1
            elif len(submitted) != len(case["specs"]):
                time.sleep(.001)
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
    result = ledger.summary()
    idle = upstream_idle(engine)
    result.update(wall_s=wall, tokens_per_second=result["actual_output_tokens"] / wall, steps=steps,
        packed_verify_calls=0, verified_request_participations=0, effective_output_block_tokens=0,
        mean_effective_output_block_tokens_per_packed_verify_call=None,
        mean_effective_output_block_tokens_per_verified_request=None,
        peak_used_pages=allocation.peak_blocks,
        peak_reserved_kv_slots=allocation.peak_blocks * engine.scheduler.block_manager.block_size,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        resident_gpu_allocated_bytes=resident_allocated, resident_gpu_reserved_bytes=resident_reserved,
        peak_gpu_allocated_delta_bytes=torch.cuda.max_memory_allocated() - resident_allocated,
        peak_gpu_reserved_delta_bytes=torch.cuda.max_memory_reserved() - resident_reserved,
        gpu_allocated_bytes_at_finish=torch.cuda.memory_allocated(),
        gpu_reserved_bytes_at_finish=torch.cuda.memory_reserved(),
        native_peak_running_requests=peak_running, native_peak_waiting_requests=peak_waiting,
        native_running_capacity_semantics="original scheduler unchanged; configured max_num_seqs limits scheduled sequences, not necessarily total running",
        before_cleanup=idle, after_cleanup=dict(idle))
    return result


def environment(torch):
    packages = {}
    for name in ("torch", "transformers", "triton", "numpy", "flash-attn", "jetspec"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
            "packages": packages}


def worker(args):
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch, nanovllm
    from nanovllm.layers import attention
    official_jetspec = None
    if args.mode == "jetspec":
        import jetspec as official_jetspec
    require(torch.cuda.is_available(), "CUDA is required")
    require(Path(nanovllm.__file__).resolve().is_relative_to(Path(args.repo).resolve()), "import escaped clean repo")
    initial = source_identity(nanovllm, official_jetspec)
    expected = UPSTREAM_HEAD if args.mode == "upstream" else FINAL_HEAD
    require(initial["git"]["head"] == expected and initial["git"]["status_porcelain"] == "",
            "wrong or dirty production snapshot")
    if args.mode == "jetspec":
        require(initial["production_sha256"] == FINAL_PRODUCTION_SHA256,
                "Final source fingerprint differs from the frozen matched benchmark")
    manifest = load_manifest(args.manifest)
    mode = "upstream_flashattn" if args.mode == "upstream" else "jetspec"
    config = dict(max_num_seqs=8, max_model_len=4096, max_num_batched_tokens=4096,
                  gpu_memory_utilization=.8, tensor_parallel_size=1, enforce_eager=True, kvcache_block_size=256)
    report = {"schema_version": 1, "status": "in_progress", "mode": mode,
        "source": initial, "environment": environment(torch), "config": config,
        "invocation": {"argv": sys.argv, "arguments": vars(args)},
        "models": {"target": previous._helpers.checkpoint(args.target)},
        "manifest_path": str(Path(args.manifest).resolve()), "manifest_file_sha256": file_sha(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "cases": [{k: v for k, v in case.items() if k != "specs"} for case in manifest["cases"]],
        "policy": {"warmup_per_case": args.warmup, "timed_repeats_per_case": args.repeats,
            "allocator_start": "new native BlockManager before every sample; within-sample prefix caching unchanged",
            "mode_isolation": "different worker processes/environments; baseline has no Draft weights",
            "timed_order": "all six case warmups, then each case's three timed repeats; all samples retained",
            "sample_clock": "offered arrival wall clock, production stepping, token ledger, final synchronization; setup/cleanup excluded",
            "max_num_seqs": "case concurrency, set on existing native config and scheduler",
            "initialization_max_num_seqs": 8,
            "representative_case_preregistered": {"concurrency": 8, "output_cap_scale": 512},
            "greedy_temperature": 0, "ignore_eos": True,
            "enable_chunked_prefill": False if mode == "jetspec" else "not an upstream configuration option",
            "optimization": "serving" if mode == "jetspec" else "pristine upstream",
            "delivery_observation": "original step-return Sequence token availability (read-only refs); no native public token stream"
            if args.mode == "upstream" else "existing public exactly-once token events",
            "no_production_edits_or_parameter_tuning": True,
            "gpu_peak_memory": "PyTorch allocated/reserved high-water marks, not NVML total-device peaks"},
        "warmups": [], "samples": [], "source_frozen_checks": []}
    engine = runtime = None
    save(args.output, report)
    try:
        if args.mode == "upstream":
            import flash_attn
            require(attention.flash_attn_varlen_func is flash_attn.flash_attn_varlen_func and
                    attention.flash_attn_with_kvcache is flash_attn.flash_attn_with_kvcache,
                    "native attention does not bind the original FlashAttention APIs")
            require("scaled_dot_product_attention" not in Path(attention.__file__).read_text(),
                    "pristine upstream attention unexpectedly contains an SDPA path")
            # This exercises the pristine public sampling constructor, not a workaround.
            require(nanovllm.SamplingParams(temperature=0).temperature == 0, "native greedy unavailable")
        else:
            require(attention.flash_attn_varlen_func is None and attention.flash_attn_with_kvcache is None,
                    "frozen JetSpec environment/backend unexpectedly changed")
        report["attention_backend"] = {"module_path": str(Path(attention.__file__).resolve()),
            "module_sha256": file_sha(attention.__file__),
            "flash_attn_varlen_func_available": callable(attention.flash_attn_varlen_func),
            "flash_attn_with_kvcache_available": callable(attention.flash_attn_with_kvcache),
            "original_flash_package_bindings": args.mode == "upstream",
            "sdpa_source_path_present": "scaled_dot_product_attention" in Path(attention.__file__).read_text()}
        torch.manual_seed(0)
        engine = nanovllm.LLM(args.target, **config)
        require({str(p.dtype) for p in engine.model_runner.model.parameters()} == {"torch.bfloat16"},
                "Target parameter dtype is not BF16")
        pool = engine.model_runner.kv_cache
        require(pool.dtype == torch.bfloat16, "KV dtype is not BF16")
        report["pool"] = previous.pool_identity(pool)
        report["pool"]["block_size"] = engine.scheduler.block_manager.block_size
        report["target_parameter_dtypes"] = ["torch.bfloat16"]
        if args.mode == "jetspec":
            runtime = engine.get_jetspec_batch_runtime(args.draft)
            require(runtime.target is engine.model_runner.model and runtime.kv_pool is pool,
                    "Final JetSpec escaped its runner Target/KV")
            report["models"]["draft"] = previous._helpers.checkpoint(args.draft)
            report["draft_residency"] = previous.draft_identity(runtime)
            report["resolved_jetspec_policy"] = {"tree_depth": 15, "tree_width": 7,
                "max_tree_budget": 63, "default_tree_budget": 63, "max_admissions_per_step": 2,
                "max_prefill_tokens": 4096, "max_verify_tokens": 4096,
                "enable_chunked_prefill": False, "optimization": "serving", "record_timing": False}
        else:
            report["draft_residency"] = {"loaded": False}
        jobs = [("warmup", case, i) for case in manifest["cases"] for i in range(args.warmup)]
        jobs += [("timed", case, i) for case in manifest["cases"] for i in range(args.repeats)]
        save(args.output, report)
        for index, (phase, case, repeat) in enumerate(jobs):
            before = source_identity(nanovllm, official_jetspec)
            require(before == initial, "source changed before sample")
            if args.mode == "upstream":
                require(attention.flash_attn_varlen_func is flash_attn.flash_attn_varlen_func and
                        attention.flash_attn_with_kvcache is flash_attn.flash_attn_with_kvcache,
                        "original FlashAttention bindings changed")
            cold = prepare_upstream(engine, case["concurrency"]) if args.mode == "upstream" else \
                previous.prepare_sample(engine, runtime, args.draft, case["concurrency"], "jetspec")
            pool_now = previous.pool_identity(pool)
            require(all(report["pool"][key] == value for key, value in pool_now.items()), "KV pool changed")
            require(sys.getprofile() is None, "a profiler is active before sample")
            print(f"[{index + 1}/{len(jobs)}] {mode} {phase} c{case['concurrency']} O{case['output_cap_scale']} repeat={repeat + 1} START", flush=True)
            if args.mode == "upstream" and phase == "warmup" and index == 0:
                with FlashCallEvidence(attention) as evidence:
                    sample = serve_upstream(engine, case, args.deadline)
                report["flash_attention_actual_path"] = evidence.result()
            else:
                sample = serve_upstream(engine, case, args.deadline) if args.mode == "upstream" else \
                    previous.serve(engine, runtime, case, "jetspec", args.deadline)
            require(sys.getprofile() is None, "qualification profiler leaked into timed path")
            sample.update(phase=phase, mode=mode, repeat=repeat, case_index=case["case_index"],
                concurrency=case["concurrency"], output_cap_scale=case["output_cap_scale"],
                workload_sha256=case["workload_sha256"], cold_allocator=cold,
                pool=report["pool"], profiler_active_during_timed=False if phase == "timed" else None)
            require(source_identity(nanovllm, official_jetspec) == initial, "source changed during sample")
            report["source_frozen_checks"].append({"job": index, "before_after_equal": True})
            report["warmups" if phase == "warmup" else "samples"].append(sample)
            save(args.output, report)
            print(f"[{index + 1}/{len(jobs)}] DONE tokens={sample['actual_output_tokens']} wall={sample['wall_s']:.3f}s tok/s={sample['tokens_per_second']:.3f} pages={sample['peak_used_pages']}", flush=True)
        require(len(report["warmups"]) == 6 * args.warmup and len(report["samples"]) == 6 * args.repeats,
                "worker sample matrix incomplete")
        report["source_end"] = source_identity(nanovllm, official_jetspec)
        require(report["source_end"] == initial, "production snapshot changed at completion")
        require(file_sha(args.manifest) == report["manifest_file_sha256"], "manifest changed during benchmark")
        report.update(status="complete", passed=True)
        save(args.output, report)
    except BaseException as error:
        report.update(status="failed", passed=False,
                      failure={"type": type(error).__name__, "message": str(error)})
        save(args.output, report)
        raise
    finally:
        if engine is not None:
            atexit.unregister(engine.exit)
            engine.exit()


def combined_summary(upstream_samples, jetspec_samples):
    # Only the established metric formulas are reused; this is NOT the prior
    # backend-matched experiment and no temporal paired-repeat claim is made.
    proxy = [{**sample, "mode": "ordinary_ar"} for sample in upstream_samples]
    proxy += [{**sample, "mode": "jetspec"} for sample in jetspec_samples]
    summary = previous.matched_summary(proxy)
    for case in summary["cases"]:
        case["jetspec_over_upstream_median_throughput_ratio"] = case.pop("jetspec_over_ar_median_throughput_ratio")
        case.pop("paired_repeat_throughput_ratios")
        case.pop("median_paired_repeat_throughput_ratio")
        for variant in case["variants"]:
            variant["mode"] = "upstream_flashattn" if variant["mode"] == "ordinary_ar" else "jetspec"
            samples = upstream_samples if variant["mode"] == "upstream_flashattn" else jetspec_samples
            runs = [s for s in samples if (s["concurrency"], s["output_cap_scale"]) ==
                    (case["concurrency"], case["output_cap_scale"])]
            variant["raw_throughputs_tokens_per_second"] = [s["tokens_per_second"] for s in runs]
            variant["max_observed_per_request_delivery_gap_s"] = max(
                s["per_request_delivery_gap_distribution_s"]["max"] for s in runs)
            variant["all_allocator_cleanup_passed"] = all(s["after_cleanup"]["used_blocks"] == 0 and
                s["after_cleanup"]["free_blocks"] == s["pool"]["shape"][2] for s in runs)
    summary["comparison"] = "practical system-to-system; independent worker processes/environments and natural KV pools, identical offered token manifest/user configuration"
    summary["repeat_pairing"] = "No temporal pairing between separate workers; speedup is the ratio of the three-run throughput medians."
    return summary


def combine(args):
    certificate = json.loads(Path(args.flash_validation).read_text())
    require(certificate["status"] == "passed" and certificate["varlen_callable"] and
            certificate["kvcache_callable"] and all(case["gpu_call_passed"] for case in certificate["cases"]),
            "FlashAttention independent GPU validation did not pass")
    require({case["api"] for case in certificate["cases"]} ==
            {"flash_attn_varlen_func", "flash_attn_with_kvcache"}, "GPU certificate missed one Flash API")
    upstream, jet = [json.loads(Path(path).read_text()) for path in (args.upstream_results, args.jetspec_results)]
    require(upstream["passed"] and jet["passed"] and upstream["status"] == jet["status"] == "complete",
            "refusing to publish incomplete/failed benchmark")
    require(upstream["mode"] == "upstream_flashattn" and jet["mode"] == "jetspec", "worker modes swapped")
    for key in ("manifest_sha256", "manifest_file_sha256", "config", "cases"):
        require(upstream[key] == jet[key], f"workers do not match {key}")
    require(upstream["models"]["target"] == jet["models"]["target"], "target checkpoint/tokenizer differs")
    for key in ("torch", "cuda", "gpu", "compute_capability"):
        require(upstream["environment"][key] == jet["environment"][key], f"worker environment differs: {key}")
    require(upstream["source"]["git"]["head"] == UPSTREAM_HEAD and jet["source"]["git"]["head"] == FINAL_HEAD,
            "published snapshots differ from preregistered source revisions")
    require(upstream["flash_attention_actual_path"]["sdpa_calls"] == 0 and
            upstream["flash_attention_actual_path"]["profiler_removed_for_timed_samples"], "Flash path proof failed")
    require(jet["policy"]["enable_chunked_prefill"] is False, "Final JetSpec unexpectedly enables chunking")
    require((certificate["torch"], certificate["cuda"], certificate["gpu"]) ==
            tuple(upstream["environment"][key] for key in ("torch", "cuda", "gpu")),
            "Flash GPU certificate used a different device/software")
    summary = combined_summary(upstream["samples"], jet["samples"])
    require(len(summary["cases"]) == 6, "comparison matrix incomplete")
    representative = next(case for case in summary["cases"] if
                          (case["concurrency"], case["output_cap_scale"]) == (8, 512))
    report = {"schema_version": 1, "status": "complete", "passed": True,
        "benchmark_type": "pristine upstream FlashAttention vs frozen Final JetSpec practical system comparison",
        "upstream_repository": "https://github.com/GeeeekExplorer/nano-vllm",
        "baseline_revision_selection": {"preferred_commit": "bb823b3e06983d71485a8e1f23715ebd87d98ef8",
            "selected_commit": UPSTREAM_HEAD,
            "reason": "preferred project-start commit rejects greedy; selected closest preceding official ancestor that still has native greedy support; no upstream code modifications",
            "not_claiming_selected_commit_is_bb823b3": True},
        "prior_comparison_A": {"label": "fork ordinary AR eager/SDPA vs Final JetSpec matched benchmark",
            "representative_c8_output512_speedup": 7.27, "is_new_measurement": False},
        "comparison_B": {"label": "pristine upstream df99418 FlashAttention vs Final JetSpec b388330",
            "representative_case_preregistered": {"concurrency": 8, "output_cap_scale": 512},
            "representative_median_speedup": representative["jetspec_over_upstream_median_throughput_ratio"]},
        "artifacts": [{"mode": worker["mode"], "path": str(Path(path).resolve()), "sha256": file_sha(path)}
            for worker, path in ((upstream, args.upstream_results), (jet, args.jetspec_results))],
        "flash_attention_gpu_validation": {"path": str(Path(args.flash_validation).resolve()),
            "sha256": file_sha(args.flash_validation), "evidence": certificate},
        "upstream_worker": upstream, "jetspec_worker": jet, "summary": summary,
        "interpretation": "The output scales are maximum caps in the original mixed-cap two-wave workload, not uniform output lengths. Different layouts/backends need not produce bitwise-equivalent greedy tokens; exact caps/delivery and existing numerical contract are retained. This two-arm comparison measures combined speculative+serving gains, not attribution percentages."}
    save(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("upstream", "jetspec", "combine"), required=True)
    parser.add_argument("--repo")
    parser.add_argument("--target")
    parser.add_argument("--draft")
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--upstream-results")
    parser.add_argument("--jetspec-results")
    parser.add_argument("--flash-validation")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--deadline", type=float, default=1800)
    args = parser.parse_args()
    if args.mode == "combine":
        require(args.upstream_results and args.jetspec_results and args.flash_validation,
                "combine requires both complete worker artifacts and independent Flash GPU certificate")
        combine(args)
    else:
        require(args.repo and args.target and args.manifest and (args.mode == "upstream" or args.draft),
                "worker requires clean repo/checkpoints/immutable manifest")
        require(args.warmup >= 1 and args.repeats >= 3 and args.deadline > 0, "need warmup >=1, repeats >=3")
        worker(args)


if __name__ == "__main__":
    main()
