#!/usr/bin/env python3
"""One useful four-round launch profile; never formal throughput evidence."""
import argparse
from types import SimpleNamespace

import jetspec_tree_serving_profile as profile
import jetspec_official_port_benchmark as benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "target", "draft", "manifest", "output", "expected-head", "expected-production-sha"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--target-execution", choices=("eager", "cuda_graph"), default="cuda_graph")
    parser.add_argument("--target-kernels", choices=("reference", "fused_rope", "fused_rope_gqa"), default="fused_rope")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-scale", type=int, default=512)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--deadline", type=float, default=600)
    args = parser.parse_args()
    args.skip_stage = True
    original = profile.serving.prepare_candidate
    def prepare(engine, runtime, draft, concurrency, backend):
        profile.require(backend == "sdpa", "profile must retain qualified Draft/prefill policy")
        return benchmark.prepare_jetspec(engine, runtime,
            SimpleNamespace(draft=draft, target_execution=args.target_execution,
                            target_kernels=args.target_kernels), concurrency)
    try:
        profile.serving.prepare_candidate = prepare
        profile.run(args)
    finally:
        profile.serving.prepare_candidate = original
    # Bind the wrapper's policy/source too: the inherited profiler records all
    # original trace/production hashes and marks every measurement diagnostic.
    report = benchmark.json.loads(benchmark.Path(args.output).read_text())
    report["official_port_execution_policy"] = {
        "target_execution": args.target_execution, "target_kernels": args.target_kernels}
    report["official_port_profile_harness_sha256"] = benchmark.upstream.file_sha(__file__)
    trace = benchmark.json.loads(benchmark.Path(report["kernel_window"]["trace_path"]).read_text())
    graph_launches = profile.Counter(event.get("name", "") for event in trace.get("traceEvents", [])
        if event.get("ph") == "X" and event.get("cat") in ("cuda_runtime", "cuda_driver")
        and "GraphLaunch" in event.get("name", ""))
    report["kernel_window"]["cuda_graph_launch_calls"] = dict(graph_launches)
    report["kernel_window"]["cuda_graph_launch_count"] = sum(graph_launches.values())
    report["kernel_window"]["launch_count_note"] = (
        "Individual kernel host launches and graph host launches are separate. "
        "CUDA Graph replay does not remove the underlying GPU kernels; their activity is recorded by CUPTI.")
    benchmark.save(args.output, report)


if __name__ == "__main__":
    main()
