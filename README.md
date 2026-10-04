<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

## Experimental JetSpec integration

`LLM.generate_jetspec(prompt, draft_model, tree_backend="paged")` is an opt-in
greedy Qwen3 path. It currently requires one request, TP=1, eager execution,
depth=15, width=7 and a 63-node tree. The ordinary scheduler and attention paths
are unchanged; this is not yet Continuous Batching or prefix-cache integration.

The Phase 3.0 paged lifetime contract is:

```text
reserve canonical tail + acquire reusable scratch
  -> verify tree in scratch -> accept root-inclusive path
  -> copy only accepted raw post-RoPE K/V to canonical tail
  -> publish tokens/features/KV length -> retire scratch
```

Only canonical pages belong to a request's committed history. Scratch has its
own lease, never enters prefix hashing, and is reused only after a CUDA event
orders the previous verify/copy. Unused destination reservations are returned;
finish, abort and exceptions fence GPU work before releasing pages. At each
completed round, `KV length == feature length == committed tokens - 1`:
the correction token remains an uncached anchor. Commit copies O(accepted path)
KV, not the full history; tapped feature concatenation is still history-sized.

Run the small-tensor tests without model weights:

```bash
python -m unittest discover -s tests -v
```

The CPU suite replays 10/100/1000 rounds. Additional CUDA stream tests use small
KV tensors and skip when CUDA is unavailable. Verify and commit must use the
active lease's stream; between-round stream handoff waits on its readiness event.

`benchmarks/jetspec_phase3.py --help` describes the matched baseline/new GPU
comparison and separate oracle, raw-byte and poison/reuse checks. Report leased
committed/scratch/destination slots separately from the globally preallocated
GPU pool: lower lease amplification does not imply the entire pool shrinks.
The legacy runtime remains c1-only. The separate packed runner below does not
call or change that runtime.

### Phase 3.1: packed ragged verification

```python
result = llm.generate_jetspec_batch(
    ["First prompt", "Second prompt", "Third prompt"], draft_model,
    max_tokens=[32, 48, 32], tree_budgets=[63, 31, 47],
)
```

This is an experimental greedy Qwen3, TP=1, eager path, not the ordinary
Continuous Batching scheduler. It performs one unpadded Target forward over
`Q = sum(tree_sizes)`; prefill and Draft proposals are still request-local.
`get_jetspec_batch_runtime()` additionally exposes `create_request`, `step`,
`finish` and `cancel` for admission, removal and reordering between rounds.
Ordinary generation cannot interleave live packed requests. It releases an
idle runner's scratch before using the shared allocator.

Each `JetSpecRequest` owns canonical prefix pages, tokens, tapped features, an
independent Draft cache, output limits and acceptance history. Packed metadata
contains query offsets, query-to-request/local-row maps, per-request prefix
lengths/page tables, flat scratch slots, and concatenated request-local
ancestor biases (`sum(T_i**2)`, not a `Q**2` inter-request mask). RoPE positions
are `prefix_length_i + depth_i`, never global packed query indices. Each
attention program resolves only its own request's prefix and tree range.
Both c1 and packed kernels use the explicit FP32/TILE=64 online-softmax path;
increasing concurrency never silently selects a different BF16 kernel.

One runner-owned `TreeScratchArena` grows to the admitted batch high-water
mark (`ceil(Q / 256)` pages), then reuses those pages across rounds and request
lifetimes. A smaller batch does not force shrinking. Individual request
cleanup never frees that arena; runner close or explicit idle release does.
CUDA retirement events order reuse, including cross-stream handoff. Private
pages do not enter prefix hashing; prefix sharing is not yet integrated.

`BatchTreeTransaction` reserves all worst-path canonical destination growth
and missing arena capacity atomically. Acceptance is independent per request.
All paths, token lists and features are prepared before one accepted-only
all-layer physical KV copy; publication then updates every request. EOS and
output caps truncate both emission and the cached path, preserving
`KV length == feature length == committed tokens - 1`. Precommit failure
preserves all old prefixes. After the irreversible commit boundary, reporting
failure retains the new prefixes and idempotently reconciles output records;
it must not be interpreted as permission to replay the committed round.

Qualification commands:

```bash
RUN_JETSPEC_PACKED_GPU_TESTS=1 python -m unittest discover -s tests -v
python benchmarks/jetspec_phase31.py --mode batch --repo "$PWD" \
  --output /tmp/jetspec-phase31.json --warmup 1 --repeats 3
```

The harness separates timed c1/c2/c4 runs from full-layer byte checks, scratch
poison, historical/rejected-node isolation, same-total-Q request/branch
perturbations, and exception/recovery tests. It fingerprints production files,
including new files, before and after testing. The real 256/512-output sanity
uses the pinned Phase 3.0 commit and a matched resident-model baseline.

Important numerical qualification limit: frozen request/branch isolation and
raw KV commit checks pass, but multi-request closed-loop token sequences do
not all match a separately executed packed AR comparator. Observed first
divergences include BF16 near ties even with the same total Q; tree versus
chain reduction grouping and changing request-removal/GEMM shapes are not
bitwise-equivalent references. The harness reports exact-match failures and
first-divergence margins explicitly. These tests do not establish strict
AR-token equivalence for all serving schedules. The identical-state diagnostic
below qualifies the six failures as numerical drift, not a semantic/addressing
fix. Strict cross-layout lossless decoding remains an unsupported guarantee.

[Recorded qualification](benchmarks/phase31_qualification.json) contains source
fingerprints, first-divergence evidence, timing distributions and raw-artifact
locations. On RTX 5090 / BF16 Qwen3-8B (eager TP=1, warmup=1, three repeats),
aggregate ordinary/packed throughput medians were 34.55/88.60 tok/s for c2
ragged and 48.21/139.47 tok/s for c4 ragged. c1 remained 5/5 oracle-exact;
the separate packed root-AR comparison was 14/20 exact, **not qualified** as a
strict lossless multi-request gate. All 302 frozen isolation records, 99
all-layer copy checks, 179 history/rejected checks and real API/cleanup checks
passed. A small-tensor 1000-round replay at c1/c2/c3/c4 retained one shared
scratch page, not one page per request or per historical tree.

Request-local state, packed metadata, shared scratch and step-boundary
admission are foundations for Continuous Batching. Scheduler integration,
batched/chunked prefill and Draft, backpressure/preemption, prefix sharing,
asynchronous serving, TP and CUDA graphs remain separate work.

### Phase 3.1 numerical qualification

The follow-up [numerical qualification](benchmarks/phase31_numerical_qualification.json)
supplements, rather than rewrites, the original 14/20 strict-match result.
Production arithmetic, precision flags and serving code are unchanged.

```bash
PYTHONFAULTHANDLER=1 python benchmarks/jetspec_phase31_numerics.py \
  --original /root/autodl-tmp/benchmarks/jetspec-phase31/final-stable.json \
  --output /tmp/jetspec-phase31-numerics.json
```

The diagnostic snapshots actual canonical KV before commit, restores it
byte-for-byte, and teacher-forces the same ancestor tokens through serial
root-AR forwards. Instrumentation observes the actual 36-layer Target:
input hidden, normalization, split Q/K/V, RoPE, attention, o_proj, residuals,
MLP, final hidden and lm_head. Restored packed logits must reproduce the
original logits exactly. This avoids comparing two already-drifted caches
and mistaking their downstream differences for a new correctness bug.

Zero-based first materialized BF16 differences, starting from identical KV:

| Case / prompt | Disputed output index | First BF16 stage | Tree node |
| --- | ---: | --- | ---: |
| c2 equal / natural language | 14 | layer 1 attention output | 4 |
| c2 equal / math | 27 | layer 2 attention output | 8 |
| c3 ragged 63/31/47 / math | 27 | layer 13 attention output | 8 |
| c4 equal / natural language | 14 | layer 2 attention output | 4 |
| c4 ragged / math | 27 | layer 2 attention output | 8 |
| c4 ragged / long continuation | 26 | layer 14 attention output | 2 |

Each first BF16 difference affects only 1–2 of 4096 elements. Identical Q
and chronological visible K/V are checked explicitly. An independent CPU
FP64 oracle reconstructs visibility from parent links, not the production
mask. The actual TILE=64 Triton kernel is replayed with FP32 output, both
with sparse tree-key positions and compact chronological keys; rounding must
reproduce the actual packed and serial BF16 outputs respectively. This
separates pre-round reduction drift from its later BF16 manifestation.
All six first FP32 differences are at layer 0 attention, at the same listed
nodes (max-absolute delta 1.49e-8–5.96e-8). These round-0 origins predict
output indices 2/3, before the disputed outputs in the table. All 36 layers
of earlier accepted nodes remain FP32-equal. Across 364 sparse/compact
operator controls, the worst FP64-oracle error is 1.21e-5 absolute and
8.32e-7 relative RMS, below the predeclared bounds.

At the disputed outputs, five traces first differ in attention over already
numerically different historical KV. Long continuation additionally has
identical layer-0 projection input but Q=126 versus Q=204; vanilla `F.linear`
reproduces both Q outputs without any JetSpec attention or slot addressing.
Sharing the packed round-start KV removes all six disputed argmax flips.
Full token prefixes/positions, independent ancestry, fixed-layout layerwise
request/branch perturbations and raw commit checks pass. Near-tie margins
explain the flips only alongside these operator controls; the inequality
`margin <= 2 * ||logits_a - logits_b||_infinity` is not itself a correctness
proof.

Phase 3.1 is frozen under this bounded, finite-input numerical contract:

- Semantic gates remain exact: request-local prefix/ancestry/depth/RoPE,
  slot ownership, accepted-only byte-exact commit, immutable history,
  transaction recovery and scratch retirement/reuse.
- On the same hardware, software versions and precision flags, repeated
  identical-state executions with the same shape/layout, and
  finite other-request/off-branch perturbations at fixed layout, must retain
  bitwise-identical selected results. Arbitrary NaN/Inf activations in valid
  masked nodes are not covered (`0 * NaN` is not zero).
- Pre-BF16 FP32 attention is compared with independent FP64 visible-key
  attention: scaled max-absolute error and relative RMS must both be <=1e-4.
  BF16 quantization is reported separately, not used to loosen this bound.
  The isolated BF16 Q GEMM uses a separate predeclared scaled/RMS envelope
  of 2^-7; this is an empirical qualification envelope, not a universal
  dot-product error theorem or a per-element one-ULP guarantee.
- Different GEMM shapes or tree/chain key layouts are numerically qualified,
  not required to have bitwise-identical logits or complete greedy token
  sequences. The strict 14/20 result remains visible; exact cross-schedule
  AR tokens and arbitrary sampling-distribution equivalence are not claimed.

An exit-139 retry with the original harness completed with exit 0; the six
token mismatches were reproduced, but the native crash was not. It remains
an unlocalized known anomaly. These findings unblock subsequent scheduler
work under the stated contract; no Continuous Batching functionality is
implemented in this numerical qualification.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
