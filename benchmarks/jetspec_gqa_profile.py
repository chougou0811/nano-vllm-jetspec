#!/usr/bin/env python3
"""Three fixed output512 cases: one diagnostic run each, not formal throughput.

Each case warms its own exact-shape graph cache. Stage CUDA events cover the
whole run; CUPTI captures only four real late verification rounds. No extra
correctness cases, no kernel parameter search, no timings reused as E2E medians.
"""
import argparse
import atexit
import json
from pathlib import Path
import sys

import jetspec_official_port_benchmark as b
import jetspec_tree_serving_profile as profile
import jetspec_phase4 as stages


def run(args):
    sys.path.insert(0,args.repo)
    import torch,nanovllm,jetspec
    source=b.source_identity(nanovllm,jetspec)
    b.check_source(source,args.expected_head,args.expected_production_sha)
    report={'status':'in_progress','diagnostic_only':True,'source':source,
            'harness_sha256':b.upstream.file_sha(__file__),
            'policy':{'target_execution':args.target_execution,'target_kernels':args.target_kernels},
            'scope':'one diagnostic execution per case; events include host launch gaps; kernel busy time excludes them',
            'cases':[]}
    manifest=b.upstream.load_manifest(args.manifest)
    cases=b.select_cases(manifest,'c1_o512,c4_o512,c8_o512')
    engine=None
    try:
        torch.manual_seed(0)
        engine=nanovllm.LLM(args.target,**b.policy.USER_CONFIG)
        runtime=engine.get_jetspec_batch_runtime(args.draft)
        for case in cases:
            if runtime._target_graph is not None:
                runtime._target_graph.close();runtime._target_graph=None
            b.prepare_jetspec(engine,runtime,args,case['concurrency'])
            print('profile warmup',case['concurrency'],flush=True)
            warm=b.previous.serve(engine,runtime,case,'jetspec',args.deadline)
            b.prepare_jetspec(engine,runtime,args,case['concurrency'])
            trace=Path(args.output).with_suffix(f'.c{case["concurrency"]}.trace.json')
            print('profile diagnostic',case['concurrency'],flush=True)
            graph_before=b.graph_snapshot(runtime)
            with stages.StageProbe(engine) as stage, profile.TargetRanges(engine), \
                 profile.LateWindow(engine,128,4,trace) as window:
                result=b.previous.serve(engine,runtime,case,'jetspec',args.deadline)
            kernel=window.summary()
            raw=json.loads(trace.read_text())
            graph_launches=profile.Counter(e.get('name','') for e in raw.get('traceEvents',[])
                if e.get('ph')=='X' and e.get('cat') in ('cuda_runtime','cuda_driver') and 'GraphLaunch' in e.get('name',''))
            kernel['graph_launch_calls']=dict(graph_launches)
            attention=kernel['attention_cuda_activity']
            kernel['tree_attention_kernel_calls']=sum(e['calls'] for e in attention)
            kernel['tree_attention_kernel_gpu_ms']=sum(e['total_gpu_ms'] for e in attention)
            kernel['mean_packed_tree_kernel_ms']=kernel['tree_attention_kernel_gpu_ms']/kernel['tree_attention_kernel_calls']
            row={'concurrency':case['concurrency'],'output_cap_scale':512,
                'warmup':warm,'diagnostic_run':result,'stages':stage.metrics(),
                'kernel_window':kernel,'graph_delta':b.graph_delta(graph_before,b.graph_snapshot(runtime))}
            report['cases'].append(row)
            b.save(args.output,report)
        report.update(status='complete',passed=True,source_end=b.source_identity(nanovllm,jetspec))
        b.require(report['source_end']==source,'profile source changed')
        b.save(args.output,report)
    except BaseException as e:
        report.update(status='failed',passed=False,failure=str(e));b.save(args.output,report);raise
    finally:
        if engine:
            atexit.unregister(engine.exit);engine.exit()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','target','draft','manifest','output','expected-head','expected-production-sha'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--target-execution',default='cuda_graph')
    p.add_argument('--target-kernels',choices=('fused_rope','fused_rope_gqa'),required=True)
    p.add_argument('--deadline',type=float,default=600)
    run(p.parse_args())
