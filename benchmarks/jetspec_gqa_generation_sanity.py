#!/usr/bin/env python3
"""Three fixed 32-token samples and one same-state Target comparison only."""
import argparse
import atexit
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys

import jetspec_official_port_benchmark as benchmark


def run(args):
    sys.path.insert(0,args.repo)
    import torch,nanovllm,jetspec
    torch.manual_seed(0)
    manifest=benchmark.upstream.load_manifest(args.manifest)
    case=next(c for c in manifest['cases'] if (c['concurrency'],c['output_cap_scale'])==(8,512))
    prompts=[case['specs'][i]['prompt'][:n] for i,n in ((0,128),(8,257),(9,1024))]
    report={'status':'in_progress','source':benchmark.source_identity(nanovllm,jetspec),
        'scope':'three manifest-derived prompts, 32 tokens each; no exhaustive lifecycle suite',
        'generation_gate':'aggregate token LCS >=90%; each request >=80%; basic consistency, not bitwise equivalence',
        'samples':{}}
    def save():
        benchmark.save(args.output,report)
    engine=None
    try:
        engine=nanovllm.LLM(args.target,**benchmark.policy.USER_CONFIG)
        runtime=engine.get_jetspec_batch_runtime(args.draft)
        for policy in ('fused_rope','fused_rope_gqa'):
            benchmark.prepare_jetspec(engine,runtime,
                type('Args',(),dict(draft=args.draft,target_execution='cuda_graph',target_kernels=policy)),3)
            requests=[runtime.create_request(p,max_new_tokens=32,tree_budget=b,ignore_eos=True,request_id=i)
                      for i,(p,b) in enumerate(zip(prompts,(63,31,47)))]
            original=runtime._verify_batch
            checked=False
            def verify(selected,trees,transaction,metadata):
                nonlocal checked
                logits,taps=original(selected,trees,transaction,metadata)
                if policy!='fused_rope_gqa' or checked:
                    return logits,taps
                checked=True
                # Graph outputs are borrowed: own the snapshot before running
                # reference. Restore candidate KV before actual accept/commit.
                own_logits,own_taps=logits.clone(),taps.clone()
                slots=torch.cat(transaction.node_slots)
                blocks,offsets=slots//runtime.block_size,slots%runtime.block_size
                candidate_kv=runtime.kv_pool[:,:,blocks,offsets].clone()
                execution=runtime._target_execution
                runtime._target_execution='eager'
                for layer in runtime.target.model.layers:
                    layer.self_attn._jetspec_tree_gqa=False
                try:
                    ref_logits,ref_taps=original(selected,trees,transaction,metadata)
                    delta=own_logits.float()-ref_logits.float()
                    winners=own_logits.argmax(-1);reference=ref_logits.argmax(-1)
                    margin=ref_logits.float().topk(2,dim=-1).values
                    flipped=winners!=reference
                    report['same_state_first_round']={
                        'prefixes':list(metadata.prefix_lengths),'nodes':list(metadata.node_counts_host),
                        'finite':bool(torch.isfinite(own_logits).all() and torch.isfinite(own_taps).all()),
                        'logit_max_abs_delta':float(delta.abs().max()),
                        'logit_relative_rms':float(delta.square().mean().sqrt()/ref_logits.float().square().mean().sqrt()),
                        'argmax_flips':int(flipped.sum()),
                        'flip_reference_top2_margins':[float(x) for x in (margin[:,0]-margin[:,1])[flipped]],
                        'row_inf_deltas_at_flips':[float(x) for x in delta.abs().amax(-1)[flipped]],
                        'note':'diagnostic only; no unchanged 1e-4/2^-6 bitwise-style numerical gate imposed'}
                    benchmark.require(report['same_state_first_round']['finite'],'nonfinite Target output')
                finally:
                    runtime.kv_pool[:,:,blocks,offsets]=candidate_kv
                    for layer in runtime.target.model.layers:
                        layer.self_attn._jetspec_tree_gqa=True
                    runtime._target_execution=execution
                save()
                return own_logits,own_taps
            runtime._verify_batch=verify
            rounds=[]
            try:
                while any(not r.finished for r in requests):
                    record=runtime.step([r for r in requests if not r.finished])
                    rounds.append({k:record[k] for k in ('node_counts','requests')})
                report['samples'][policy]={'results':[runtime.finish(r) for r in requests],
                    'rounds':rounds,'graph':runtime._target_graph.snapshot()}
            finally:
                runtime._verify_batch=original
                for r in list(runtime.requests.values()):
                    runtime.cancel(r)
                engine.disable_jetspec()
            benchmark.require(not runtime.block_manager.used_block_ids,'KV allocator did not clean up')
            save()
        comparisons=[]
        for old,new in zip(report['samples']['fused_rope']['results'],report['samples']['fused_rope_gqa']['results']):
            a,b=old['token_ids'],new['token_ids']
            matcher=SequenceMatcher(None,a,b,autojunk=False)
            prefix=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
            comparisons.append({'request_id':old['request_id'],'exact':a==b,'common_prefix_tokens':prefix,
                'token_lcs_ratio':matcher.ratio(),'baseline_text':old['text'],'candidate_text':new['text']})
        report['comparisons']=comparisons
        benchmark.require(all(c['token_lcs_ratio']>=.8 for c in comparisons) and
            sum(c['token_lcs_ratio'] for c in comparisons)/len(comparisons)>=.9,
            'obvious small-sample generation divergence')
        report.update(status='complete',passed=True,allocator_cleanup=True)
        save()
    except BaseException as e:
        report.update(status='failed',passed=False,failure=str(e));save();raise
    finally:
        if engine:
            atexit.unregister(engine.exit);engine.exit()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','target','draft','manifest','output'):
        p.add_argument('--'+name,required=True)
    run(p.parse_args())
