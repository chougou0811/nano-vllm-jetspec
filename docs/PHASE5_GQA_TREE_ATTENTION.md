# Phase 5: official-capability GQA Tree Attention migration

Production snapshot: `3654e12667e40a47f68b2b07d677a965d73bb2fb`.
Matched previous implementation: `c210e4aabe859495430a27e85a6b4b2484721f93`.
Full raw measurements, warmups, source hashes and diagnostic evidence:
[phase5_gqa_benchmark.json](../benchmarks/phase5_gqa_benchmark.json).

This is a migration/adaptation project, not a claim to have invented these
attention techniques or replicated the whole official engine.

## Official reference and scope

Official JetSpec master was verified at
`2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f` on 2026-10-06. Sources read:
[paged_tree_attn.py](https://github.com/hao-ai-lab/JetSpec/blob/2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f/jetspec/inference_engine/paged_tree_attn.py),
[paged_tree_attn_op.py](https://github.com/hao-ai-lab/JetSpec/blob/2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f/jetspec/inference_engine/paged_tree_attn_op.py),
[compiled_verify_stack.py](https://github.com/hao-ai-lab/JetSpec/blob/2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f/jetspec/inference_engine/compiled_verify_stack.py),
[graph_capture.py](https://github.com/hao-ai-lab/JetSpec/blob/2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f/jetspec/inference_engine/graph_capture.py).
The upstream MIT notice is retained in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

| Capability | Before this round | This round |
|---|---|---|
| GQA-head K/V tile reuse | Q-head-local scans | KV-head-centric program, all four Q heads share K/V |
| Multi-Q / Tensor Core tiles | Existing reference arithmetic | Four tree rows × four Q heads; QK and PV dot |
| Online softmax | Already implemented | Retained FP32 max/sum/accumulator, official BF16-P PV |
| Direct paged access | Already implemented | Retained canonical prefix pages + scratch slot mapping |
| Packed ragged verification | Already implemented | Retained, device-side ragged Q-block mapping |
| Visible-range pruning | No grouped Q-block bound | Loop stops at the last possible visible tree row |
| Prefix scalar-page addressing | Already implemented | Retained for complete prefix tiles |
| RoPE + scratch KV scatter fusion | Already implemented | Retained |
| Exact-shape reusable Target Graph | Already implemented | Retained with distinct GQA cache signature |

Not migrated: the full compiled verification stack/custom-op compile boundary,
Draft Graphs, fused projection/norm stack, B200-specific kernel tuning, true
ancestor-aware tile skipping. A compile custom-op boundary is not necessary for
our existing manually captured Target Graph; no full `torch.compile` claim is made.

## Layout, access and lifecycle

`tree_gqa.py` launches one program per `(ragged local Q block, KV head)`.
For Qwen3-8B, `GQA=4`, `Q_TILE=4`, `M=16`, `D=128`, `K_TILE=64`, four warps.
The 16 score rows are four local tree nodes × four Q heads. A `[D,64]` K tile
and `[64,D]` V tile are loaded once into that program, not once per Q head.
Both QK and PV use Tensor Cores. Softmax tracks FP32 running max, denominator
and output accumulator; probabilities are cast to BF16 before PV, as in the
official throughput kernel. No complete attention score matrix is allocated.

Flat packed Q remains `[sum(tree_nodes),32,128]`. Device binary search uses
`cu_seqlens_q[s] // Q_TILE + s` to map a Q-block to its request. The launch grid
is `sum(tree_nodes)//Q_TILE + num_requests`; it does not freeze a host-side
maximum request tree size during Graph capture. Padding/gap programs and lanes
are masked. Each request independently supplies prefix length, block table,
tree slots and ancestor-bias offset. Heterogeneous trees are one packed launch
per attention layer, not a loop of per-request Target forwards.

Prefix addresses come directly from the request block table. Provisional tree
K/V comes directly from round-lifetime scratch slots. Complete 64-token prefix
tiles reuse one scalar page lookup (page size 256); boundary/tree tiles use
masked vector addressing. There is no production contiguous-KV gather.

Parent-before-child order is already validated by packed metadata. A Q block
cannot see tree nodes after its last row, so those complete later K tiles are
not scanned. Per-row causal checks **and the actual ancestor mask** still
exclude future/sibling nodes. This is conservative visible-range pruning, not
true ancestor-aware sparse tile skipping.

Only attention evaluation and its Graph dispatch signature change. Canonical
KV ownership, transaction reservation, per-request acceptance and accepted-only
physical commit are unchanged. Scratch remains batch/runner-owned. Exact-shape
Graph staging updates ragged metadata every replay; changing kernel policy
while idle destroys the previous Graph cache. Existing fusion stays enabled.

The old default policy remains available. Explicit activation:

```python
llm.configure_jetspec(
    draft_path, optimization="serving", enable_chunked_prefill=False,
    attention_backend="sdpa", target_execution="cuda_graph",
    target_kernels="fused_rope_gqa",
)
```

## Necessary sanity only

Three CPU launch/Graph-policy checks passed. Three synthetic BF16 fixtures cover
mixed tree sizes, zero/long prefixes, page boundaries 255/256/257, noncontiguous
pages, scratch slots, ancestor masking, request poison isolation and GQA mapping.
Changing KV head 2 affects only Q heads 8–11. A captured Graph with changed
ragged counts/prefixes/mappings matches a fresh launch. All outputs are finite;
operator max absolute differences are 0.00390625/0.015625/0.015625 and relative
RMS differences are about 0.16–0.22% against the previous operator.

Three fixed manifest-derived generation probes, 32 tokens each, produced exact
baseline/candidate token sequences (96 tokens total), with allocator cleanup.
One identical-state 141-node Target round had zero argmax flips. Its final-logit
max absolute delta was **13.8125**, relative RMS **6.8461%**: layerwise BF16-P
drift can amplify, and this is **not** a full-network small-error/bitwise result.
The nine formal serving samples also happened to produce identical per-request
outputs and verification participation counts. That is an observation on this
frozen workload, not a universal model-quality or strict AR-equivalence claim.
No exhaustive correctness campaign was run.

Sanity was recorded in the development worktree before the production commit.
The publisher verifies tested kernel/model/batch/Graph module hashes against
the clean measured snapshot. Only the user's unrelated legacy single-request
`runtime.py` differs; it is not executed by these batch serving probes.

## Fixed serving comparison

RTX 5090, Torch 2.8/CUDA 12.8, Qwen3-8B BF16, TP1, greedy, max model length
4096, memory utilization 0.8. Existing two-wave prompt/arrival manifest is
unchanged; tree budgets remain 63/31/47 and output caps are mixed 512/512/256/128.
Thus “output512” is the **cap scale**, not a claim that every request emits 512.
Each case has one warmup and three unprofiled formal runs, with no discarded
samples; ratios use throughput medians. Chunked Prefill is disabled.

| Case | Upstream FA tok/s | c210e4a tok/s | New GQA tok/s | Gain vs c210e4a | New / upstream |
|---|---:|---:|---:|---:|---:|
| c1/output512 | 46.296 | 227.981 | 278.635 | +22.22% | 6.02× |
| c4/output512 | 149.902 | 381.159 | 538.718 | +41.34% | 3.59× |
| c8/output512 | 231.293 | 412.598 | 599.913 | +45.40% | 2.59× |

Delivery TPOT is `(last_delivery-first_delivery)/(output_tokens-1)`, not the
duration of an individual burst. Values below are medians of per-run request p50s.
Peak memory is the maximum PyTorch allocated high-water among three runs, **not**
whole-device NVML memory. All arms happened to allocate 249 KV pages ×256 tokens.

| Case / arm | Offered TTFT ms | E2E s | Delivery TPOT ms | Peak allocated GiB |
|---|---:|---:|---:|---:|
| c1 upstream / c210 / new | 62 / 975 / 858 | 16.63 / 3.13 / 2.61 | 32.43 / 4.21 / 3.42 | 24.15 / 26.37 / 26.34 |
| c4 upstream / c210 / new | 137 / 638 / 578 | 14.14 / 6.12 / 4.44 | 28.50 / 8.26 / 5.85 | 24.37 / 27.27 / 27.26 |
| c8 upstream / c210 / new | 201 / 1076 / 980 | 12.58 / 9.79 / 6.97 | 28.47 / 13.94 / 9.76 | 24.48 / 28.66 / 28.66 |

Important fairness limits: pristine upstream is native-greedy ancestor
`df99418f7d6ca676550f4372cdc6e1521ce8c33d`, not preferred `bb823b3` (which
removed native greedy). Its production files are untouched; both FlashAttention
APIs are exercised, and no SDPA fallback occurs. Upstream is eager; both JetSpec
arms retain Target Graphs. This is a practical system comparison, not an
algorithm-only/Graph-matched ablation. Upstream's natural scheduler limits the
scheduled batch rather than strict global residency (observed 2/8/16 running);
JetSpec retains strict admission. User-level settings match, internal policies
are not rewritten. JetSpec **does not beat upstream TTFT**, and burst delivery
p95 gaps are larger despite better TPOT; full distributions are retained in JSON.

## Kernel ablation and final profile

Three synthetic operator fixtures, each one warmup plus three 100-replay CUDA
event spans, show 9.07×/7.36×/8.82× over the old operator. These are synthetic
event-span ratios, **not trained-network kernel busy time or E2E speedup**.
On the c8 fixture, one-variable operator probes show:

- Four-Q vs one-Q grouping: 2.77×; includes Tensor Core lane utilization effects.
- Prefix scalar addressing: about 1.04×; preserved, not newly invented here.
- Official BF16-P vs FP32-P alternative: 1.81×.
- Visible-range pruning: no measurable gain with these small trees and tile64;
  the capability is present, but no performance benefit is claimed.

GQA, Tensor Core layout and probability precision changes are coupled. There
is no separate E2E attribution for each technique. Formal c1/c4/c8 throughput and
TPOT all improve versus c210e4a; peak memory is essentially unchanged.
Accepted-only copy remains small and was not optimized.

Final trained-workload profile results are embedded in the benchmark artifact:
four late verification rounds per case for CUPTI kernel busy time, and whole-run
CUDA event Target verification spans (including host launch gaps). Profiling
overhead is diagnostic only and does not enter any formal throughput sample.

The old/new four-round windows have identical packed node counts and prefix
lengths; each has 144 attention kernel calls (36 layers ×4 rounds). Kernel
latency below is their mean busy time; Target latency is the whole-run event p50.

| Case | Tree kernel old → new ms | Kernel speedup | Target verify old → new ms | Verify speedup |
|---|---:|---:|---:|---:|
| c1/output512 | 0.09513 → 0.01276 | 7.46× | 21.99 → 15.61 | 1.41× |
| c4/output512 | 0.30072 → 0.03968 | 7.58× | 52.46 → 31.07 | 1.69× |
| c8/output512 | 0.51181 → 0.06074 | 8.43× | 65.16 → 49.02 | 1.33× |

All nine formal new-kernel runs replay warmed Graphs with zero captures and zero
eager fallbacks inside their timers. Effective emitted tokens per verified
request are unchanged at 7.05/7.20/7.47; peak leased pages are 7/29/53 and
allocator cleanup passes for every arm/sample. Throughput improvement does not
come from changed output lengths, tree budgets or observed acceptance.

In the new c8 late window, Target GPU busy time is 194.12 ms: GEMMs account for
169.38 ms (~87.3%), Tree Attention for 8.75 ms (~4.5%). The serving window still
has 6,670 non-Graph host kernel-launch calls and 142 synchronization API calls,
alongside four Target Graph launches. These bounded diagnostic counts include
other serving work and profiler overhead, not just Target or an unprofiled run.
The full-run event totals also retain about 2.10 s of Draft stage time and
1.45 s of prefill; the tree-building stage includes Draft and must not be added
to it. The large host acceptance time includes waiting for asynchronous Target
work, **not** seconds of acceptance GPU computation.

The next clear hotspot is Target GEMM plus Draft/host serving overhead, not
accepted KV copy or Tree Attention. Further ancestor pruning/warp tuning is
deferred as requested; its remaining E2E ceiling is now smaller. No further
optimization was attempted after observing these results. No hardware DRAM
bandwidth or occupancy claim is made from this profiler.

## Reproduction

Use clean worktrees at the pinned commits and the recorded environment. The
existing `jetspec_official_port_benchmark.py` driver accepts `--cases
c1_o512,c4_o512,c8_o512`, `--warmup 1 --repeats 3`; JetSpec arms select
`--target-execution cuda_graph --target-kernels fused_rope` or
`fused_rope_gqa`. Use the saved immutable `phase5_workload_manifest.json`.
`jetspec_gqa_profile.py` records one diagnostic execution per requested case.
`jetspec_gqa_report.py` publishes existing raw files and checks provenance,
matrix, FlashAttention proof, policy, cleanup and workload identity without
launching additional experiments. JSON retains all three raw runs and warmups.

For a resume, describe the kernel work as an official-technique adaptation with
ragged transaction semantics preserved; do not present the full system's 2.59×
as the kernel's isolated contribution. The new kernel step raises c8 throughput
from 412.6 to 599.9 tok/s (+45.4%) in this preregistered serving case.
