#!/usr/bin/env python3
"""Three synthetic shapes only: necessary GQA/address sanity and operator timing.

These operator timings are not serving throughput or trained Target latency.
Isolation checks compare identical shapes, not cross-layout bitwise numerics.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys

import jetspec_tree_kernel_benchmark as fixture


def run(args):
    sys.path.insert(0, args.repo)
    import torch
    from nanovllm.speculative.jetspec.paged_backend import packed_tree_attention
    from nanovllm.speculative.jetspec.tree_gqa import packed_tree_attention_gqa
    shapes = [fixture.Shape('c1_long', (1024,), (63,)),
              fixture.Shape('c4_boundaries', (0, 255, 256, 257), (63, 31, 47, 63)),
              fixture.Shape('c8_ragged', (128, 1024, 2048, 0, 257, 255, 256, 2049),
                            (63, 31, 47, 63, 31, 47, 63, 31))]
    report = {'status': 'in_progress', 'source': fixture.source_identity(args.repo),
              'harness_sha256': fixture.function_identity(run)['file_sha256'],
              'sanity_scope': 'three synthetic shapes; no full-model/bitwise equivalence claim',
              'timing': 'one warmup + 3 CUDA Graph replay event spans, 100 calls/run; no preparation in timer',
              'shapes': []}
    def save():
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    def require(value, message):
        if not value:
            raise AssertionError(message)
    def timed(fn):
        fn()  # compile outside capture/timer
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = fn()
        graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(3):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                graph.replay()
            end.record(); end.synchronize()
            times.append(start.elapsed_time(end) / 100)
        return {'raw_gpu_event_ms': times, 'median_gpu_event_ms': statistics.median(times)}
    try:
        for shape in shapes:
            q,k,v,meta,ownership,slots = fixture.prepare(shape, 73)
            def old():
                return packed_tree_attention(q,k,v,meta,.125,4)
            def new(**kw):
                return packed_tree_attention_gqa(q,k,v,meta,.125,4,**kw)
            reference, candidate = old().float(), new().float()
            delta = candidate - reference
            rms = float(delta.square().mean().sqrt())
            relative = rms / max(float(reference.square().mean().sqrt()), 1e-6)
            row = {'name': shape.name, 'prefixes': shape.prefixes, 'nodes': shape.nodes,
                   'max_abs_error': float(delta.abs().max()), 'relative_rms_error': relative,
                   'finite': bool(torch.isfinite(candidate).all()),
                   'rough_operator_tolerance': {'max_abs': .025, 'relative_rms': .012}}
            report['shapes'].append(row); save()
            require(row['finite'] and row['max_abs_error'] < .025 and relative < .012,
                    'nonfinite or obvious operator discrepancy')
            if shape.name == 'c4_boundaries':
                base = new().clone()
                # Off-path leaf: root and every other request stay unaffected.
                node = ownership['tree_slots'][0][-1]
                before = v[node//256,node%256].clone()
                v[node//256,node%256].add_(17)
                poisoned = new()
                row['off_path_root_exact'] = torch.equal(base[0],poisoned[0])
                row['off_path_other_request_exact'] = torch.equal(base[63:],poisoned[63:])
                v[node//256,node%256].copy_(before)
                # Mutate all V owned by request 1; other requests must not see it.
                owned = fixture.prefix_slots(shape,ownership,1)+ownership['tree_slots'][1]
                ix=torch.tensor(owned,device='cuda')
                before=v[ix//256,ix%256].clone()
                v[ix//256,ix%256] += 9
                poisoned=new()
                row['request_isolation_exact'] = torch.equal(base[:63],poisoned[:63]) and torch.equal(base[94:],poisoned[94:])
                row['poison_affects_owner'] = not torch.equal(base[63:94],poisoned[63:94])
                v[ix//256,ix%256]=before
                # KV head 2 must affect ONLY Q heads 8..11 (32Q/8KV).
                head_before = v[:,:,2].clone()
                v[:,:,2].add_(3)
                head_poison=new()
                row['gqa_head_mapping'] = torch.equal(base[:,:8],head_poison[:,:8]) and torch.equal(base[:,12:],head_poison[:,12:]) and not torch.equal(base[:,8:12],head_poison[:,8:12])
                v[:,:,2].copy_(head_before)
                require(all(row[n] for n in ('off_path_root_exact','off_path_other_request_exact','request_isolation_exact','poison_affects_owner','gqa_head_mapping')), 'isolation/head/ancestor mismatch')
                # Reuse one graph while ragged request lengths/counts/slots change.
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    result=new()
                changed=replace(shape,prefixes=(257,0,255,256),nodes=(31,63,63,47))
                other=fixture.layout(changed,91)
                changed_meta,_=fixture.metadata(changed,other,'cuda')
                # The pool must have room for both validated random layouts.
                require(other['num_pages']<=k.shape[0],'fixture pool too small')
                for field in ('prefix_lens','node_counts','cu_seqlens_q','block_tables','tree_slots','qq_bias','qq_bias_offsets'):
                    getattr(meta,field).copy_(getattr(changed_meta,field))
                g.replay(); snapshot=result.clone()
                fresh=packed_tree_attention_gqa(q,k,v,changed_meta,.125,4)
                row['graph_dynamic_ragged_exact']=torch.equal(snapshot,fresh)
                require(row['graph_dynamic_ragged_exact'],'graph froze ragged/prefix metadata')
                # Return the fixture to its original geometry for operator timing.
                restored,_=fixture.metadata(shape,ownership,'cuda')
                for field in ('prefix_lens','node_counts','cu_seqlens_q','block_tables','tree_slots','qq_bias','qq_bias_offsets'):
                    getattr(meta,field).copy_(getattr(restored,field))
            row['baseline']=timed(old)
            row['gqa']=timed(new)
            row['kernel_event_span_speedup']=row['baseline']['median_gpu_event_ms']/row['gqa']['median_gpu_event_ms']
            if shape.name=='c8_ragged':
                row['ablations']={name:timed(lambda kw=kw:new(**kw)) for name,kw in (
                    ('gqa_only_one_query',dict(query_tile=1)),
                    ('no_prefix_scalar',dict(prefix_scalar=False)),
                    ('no_causal_tile_pruning',dict(prune=False)),
                    ('fp32_probability',dict(fp32_probability=True)))}
            save(); print(shape.name,row['kernel_event_span_speedup'],flush=True)
        torch.cuda.synchronize()
        report.update(status='complete',passed=True); save()
    except BaseException as e:
        report.update(status='failed',passed=False,failure=str(e)); save(); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True);parser.add_argument('--output',required=True)
    run(parser.parse_args())
