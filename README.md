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

## JetSpec Phase 3.2: Continuous Batching

`LLM.configure_jetspec()` routes the normal engine `add_request`, `step`,
`is_finished` and `cancel_request` API through `JetSpecScheduler`. This is
nano-vLLM's synchronous serving event loop: clients may submit or cancel
between steps, with token deltas observable at step-return boundaries. It
is not an HTTP server, background ingress thread, or concurrent GPU executor.

```python
from nanovllm import LLM, SamplingParams

llm = LLM(target_model, enforce_eager=True, tensor_parallel_size=1,
          max_num_seqs=8, max_num_batched_tokens=4096)
llm.configure_jetspec(draft_model, max_tree_budget=63)
a = llm.add_request("First prompt", SamplingParams(temperature=0, max_tokens=128),
                    request_id="a", tree_budget=63)
llm.step()
for event in llm.last_step_info["events"]:
    print(event["request_id"], event["kind"], event["token_ids"])
b = llm.add_request("Later prompt", SamplingParams(temperature=0, max_tokens=64),
                    request_id="b", tree_budget=31)
while not llm.is_finished():
    completed, num_tokens = llm.step()  # Negative emitted-token count in JetSpec mode.
    for event in llm.last_step_info["events"]:
        # kind=tokens: incremental token_ids; terminal kinds: full token_ids.
        print(event["request_id"], event["kind"], event["token_ids"])
# cancel_request(id) also works for queued, resident and suspended requests.
llm.disable_jetspec()  # Requires all requests AND terminal events to be drained.
```

The adapter owns host tickets, FIFO waiting admission, a round-robin resident
queue, and emitted-output cursors. The Phase 3.1 runtime continues to own
canonical KV, per-request Draft caches, `BatchTreeTransaction` and the shared
scratch arena. Every decode step builds one new ragged packed batch from the
selected residents, runs one Target verification, then independently accepts
and commits each request. Completed/cancelled tickets leave the queues; no
unbounded completed-request archive is retained.

Admission performs bounded, sequential dense prefill or recompute (default
two admissions per step) and leaves room for an existing resident to decode.
Prefill has a separate nonchunked token-work budget; batched/chunked prefill
and batched Draft are not implemented. The packed-query budget is distributed
fairly across eligible requests up to their individual tree caps; the final
remaining output uses a root-only tree without a Draft call.

For cached lengths `P_i`, effective node budgets `T_i`, remaining output caps
`R_i`, page size `B`, and scratch high-water pages `S`, the conservative new
page requirement is:

```text
destination growth = sum(ceil((P_i + min(T_i, 16, R_i)) / B) - owned_pages_i)
scratch growth     = max(0, ceil(sum(T_i) / B) - S)
required free      = destination growth + scratch growth
```

Here `16` is the currently supported trained head's `tree_depth + 1`.
The read-only plan is followed by the existing atomic transaction admission;
only the latter allocates pages. Under pressure the scheduler reduces tree
budgets, selects a smaller fair batch, releases excess idle scratch if
necessary, and finally preempts a lower-priority resident. Normal arrivals
do not evict a progressing resident just to enter the batch.

Preemption retains CPU committed tokens, the existing uncached anchor and
already-emitted history, resets Draft state, then releases canonical KV.
If Draft reset fails before release, the original resident still owns its KV.
Resume prefills `committed[:-1]`, rebuilds KV/features and restores the same
anchor: it neither generates a replacement anchor nor re-emits old outputs.
Numerical changes from recomputation remain subject to the Phase 3.1 contract.
Temporary external capacity shortage returns `blocked`/`blocked_reason` for
the caller to back off and retry; an intrinsically unfit singleton produces
an explicit terminal capacity error instead of a preemption loop.

EOS, `max_tokens` (including zero) and cancellation are request-local.
An emitted cursor delivers each token once. A failed packed batch terminates
its selected requests while preserving unscheduled/queued requests, and queues
published progress and terminal errors before rethrowing; a postcommit failure
is never treated as a rollback. `last_step_info` preserves the error report,
and pending notifications remain available on the next step or explicit drain.
Cancel's immediate response confirms the terminal event subsequently delivered
in the stream; consumers should not count both as two generated completions.
The blocking `generate()` convenience API requires an initially idle queue. It
raises on error/cancelled terminals or capacity backpressure, cancels its own
requests, drains their notifications and retains the original error report;
use the incremental `step()` loop when external capacity may become available.

Ordinary, explicit packed and Continuous Batching modes are mutually exclusive.
`disable_jetspec()` returns idle scratch pages while retaining immutable Draft
weights. The ordinary scheduler now also respects already-running requests
when admitting new prefills, so configured concurrency limits apply to both
benchmark modes. This stage remains greedy, eager TP=1; prefix sharing,
asynchronous execution, TP and CUDA graphs remain future work.

Qualification and matched serving benchmark:

```bash
RUN_JETSPEC_PACKED_GPU_TESTS=1 python -m unittest discover -s tests -v
PYTHONFAULTHANDLER=1 python benchmarks/jetspec_phase32.py --repo "$PWD" \
  --output /tmp/jetspec-phase32.json --concurrencies 1,2,4,8 \
  --max-tokens 128 --warmup 1 --repeats 3
```

The benchmark keeps one Target + Draft resident in both modes, uses identical
offered wall-clock arrivals with mixed prompts/tree budgets/output caps, and
records actual between-step submission and delivery times. Fixed-step
replay, EOS/cancel, finite same-shape isolation, poisoned scratch, all-layer
raw copies/history, held-page backpressure and real preemption/resume are
separate correctness runs. Stage profiling is also separate from throughput
samples. Cross-shape ordinary/JetSpec token differences are reported, not
promoted into a new strict-bitwise gate.

Measured on RTX 5090, Qwen3-8B BF16, eager TP=1 (one warmup and three
repeats per mode/case). The two waves generate 256/337/674/1348 actual tokens
for c1/c2/c4/c8, with mixed per-request caps 128/128/17/64:

| Max residents | Ordinary tok/s | JetSpec tok/s | Ratio | Offered E2E p50, ordinary / JetSpec | Peak allocated KV slots, ordinary / JetSpec |
|---|---:|---:|---:|---:|---:|
| 1 | 24.38 | 92.00 | 3.77x | 7.869 / 2.177 s | 256 / 512 |
| 2 | 33.45 | 119.51 | 3.57x | 7.746 / 2.114 s | 512 / 768 |
| 4 | 46.22 | 175.81 | 3.80x | 11.133 / 2.715 s | 1024 / 1280 |
| 8 | 55.27 | 203.20 | 3.68x | 16.495 / 4.472 s | 2048 / 2560 |

The fixed pool is the same 249 pages / 63744 slots in both modes; these
occupancy peaks are not smaller overall GPU pool allocations. Shared scratch
high-water is 1/1/1/2 pages, not one scratch arena per request. At c8 the
mean effective emission is 3.73 tokens per verified request, with mean raw
accepted Draft length 2.78. Finite same-shape c8 Q408 isolation, poisoned
scratch/raw-copy/history checks and real three-page preemption/replay pass;
all 204 requests in the 29 audited timed/qualification runs have exactly one
terminal event. Disable returns allocator occupancy to zero.

Separate profiling points to Target/GEMM, serial per-request Draft, small
kernel launch and host synchronization as optimization priorities. In the
stage-profile workload Target verify takes 0.992 s of stream elapsed, Draft
0.727 s, and accepted KV copy only 0.0036 s; these include host launch gaps
and are nested, not additive GPU busy-time percentages. Actual CUDA traces
show BF16 GEMMs as the leading kernels rather than accepted KV copy or packed
attention alone. Batched Draft, fewer host synchronizations and bounded
batched/chunked prefill are the next candidates, not implemented in this phase.

## JetSpec Phase 4: profile-driven serving optimization

`configure_jetspec(..., optimization="serving")` now enables grouped Draft,
lightweight round decisions and capacity-managed Target features. Use
`optimization="debug"` for the previous full device checks/node records.
Explicit packed runtime calls remain debug by default; the benchmark can select
each optimization independently through `runtime.configure_optimizations()`.
The synchronous scheduler, transaction publication boundary, canonical Target
KV and runner-owned scratch ownership are unchanged.

- **Grouped Draft:** compatible requests share one actual official DFlash head
  forward. Rectangular old-prefix/new-suffix buffers use independent absolute
  positions and an explicit per-row context/noise mask. Grouping bounds both
  suffix and attention-key padding inflation to 2x. Only real context KV is
  published into compact, independently allocated request caches; padding/noise
  and batch-wide backing storage never become persistent request state.
  Singletons, unsupported cache/backend types and dynamic/long RoPE variants
  retain the official serial path. This is padded grouping, not paged Draft KV.
- **Lightweight decisions:** one round-wide top-k download feeds the unchanged
  official accum-logp heap; CPU topology/page ownership directly builds packed
  metadata. One combined Target argmax download feeds independent acceptance
  walks. Full per-node diagnostics, repeated device invariant comparisons and
  per-round timing events are disabled in serving mode, not required host
  ownership/bounds checks or stream lifetime fences. Debug can force full device
  validation. The two required decision downloads and other small transfers
  remain; this is not a synchronization-free or fully GPU-resident scheduler.
- **Feature lifetime:** `target_hidden` remains the exact visible tensor prefix
  required by the official head. Geometric backing capacity permits append-only
  accepted taps, with history copies only on growth. Unpublished tails and
  rollback-aware feature plans preserve precommit/postcommit failure semantics.
  Preemption still releases features and reconstructs them on resume.
- **Prefill:** only the final prompt row is projected through `lm_head` in
  lightweight mode. Prefill/recompute otherwise remains request-local and
  nonchunked. A resumed prefix exceeding `max_prefill_tokens` still produces
  an explicit capacity error; configure this limit for the maximum recompute
  prefix. Batched/chunked prefill is not implemented by this change.

Capacity snapshots additionally expose live/reserved Target feature bytes and
logical Draft cache bytes; Draft profiling also reports backing-storage and
padding bytes. Admission still budgets physical Target KV pages, not a complete
CUDA memory model. Feature spare capacity and batched Draft transient storage
must not be mistaken for free memory.

Qualification retains the Phase 3.1/3.2 finite-input numerical contract.
Changed Draft GEMM/SDPA layouts and final-row prefill projection do not promise
cross-shape token bitwise equality. Same-layout request/padding isolation,
independent metadata reconstruction, all-layer raw KV copies, historical KV,
exactly-once delivery and allocator cleanup remain strict gates.

The matched benchmark imports clean `7bcb754` and the optimized checkout in
separate workers using the same frozen driver, checkpoint identities, offered
arrivals and 249-page pool. It covers mixed short/1024-token prompts, mixed tree
budgets/output caps, two-wave arrivals and 128/512/1024 output caps. Delivery TPOT
is measured from first to last delivered token, separately from offered-clock
and submitted-clock TTFT/E2E; stage instrumentation and CUDA traces are separate
from timed throughput samples. See [the driver](benchmarks/jetspec_phase4.py)
and [trained-model qualification](benchmarks/jetspec_phase4_qualification.py).

### Measured results

[Phase 4 evidence](benchmarks/phase4_qualification.json) records 57 timed samples,
524 requests and 142,976 emitted tokens on RTX 5090 / Qwen3-8B BF16, eager TP=1.
The baseline is **Phase 3.2 JetSpec**, not ordinary AR. Each cell is baseline →
optimized aggregate tok/s, followed by the ratio:

| Residents | Output cap 128 | Output cap 512 | Output cap 1024 |
|---|---:|---:|---:|
| c1 | 73.14 → 76.66 (1.048x) | 99.88 → 104.92 (1.051x) | 121.35 → 126.20 (1.040x) |
| c2 | 107.23 → 126.49 (1.180x) | 146.11 → 172.40 (1.180x) | 175.06 → 198.90 (1.136x) |
| c4 | 149.81 → 181.87 (1.214x) | 186.57 → 233.73 (1.253x) | 210.32 → 261.92 (1.245x) |
| c8 | 161.61 → 211.43 (1.308x) | 207.45 → 277.40 (1.337x) | 234.05 → 316.10 (1.351x) |

c4/c8 at 128/512 use three-sample medians, including an ABBA repeat sequence;
other cells are single timed samples. Every case has one warmup. Per-request
caps are mixed `[M, M, M/2, M/4]`, not M for every request. Independent c16/128
stress reached packed concurrency 16, 249.61 tok/s and clean completion of all
32 requests; it has no c16 baseline and therefore no claimed speedup.

The 512-token optimization ladder (lightweight includes final-row prefill
projection; intermediate variants have one sample per case):

| Residents | Baseline | Lightweight | + Feature storage | + Batched Draft |
|---|---:|---:|---:|---:|
| c1 | 99.88 | 100.68 | 103.05 | 104.92 |
| c2 | 146.11 | 156.43 | 157.79 | 172.40 |
| c4 | 186.57 | 202.95 | 205.03 | 233.73 |
| c8 | 207.45 | 228.59 | 230.73 | 277.40 |

At c8/512, pooled p50 offered TTFT/E2E improve from 3.886/17.680 s to
2.965/12.994 s; submitted-clock TTFT/E2E from 3.804/17.535 s to 2.890/12.862 s.
Delivery TPOT is 29.57 → 21.82 ms. Service residence is 9.006 → 6.743 s:
this excludes nonresident queue/preemption time but is **not GPU busy time**.

Separate c8/512 profiling shows Draft host/stream time 7.397/8.233 →
3.040/3.095 s, with 1,230 serial proposals replaced by 280 actual B>1 head
forwards and 79 singleton fallbacks. Mean output per verified request is
4.558 → 4.570; the gain is not a large acceptance-rate change. Feature update
copy payload falls 39.758 → 0.750 GB, but its standalone throughput benefit is
small. Commit preparation host time actually rises 0.270 → 0.315 s; accepted
KV copy remains only about 0.04 s and was not optimized.

The late four-round trace has 23,637 → 13,246 `cudaLaunchKernel` calls and
896 → 142 `cudaStreamSynchronize` calls. Explicit synchronization CPU time
falls 164.03 → 1.76 ms, **but** `cudaMemcpyAsync` CPU time rises 11.37 →
143.32 ms as pageable download waits move inside the copy API. This is fewer
boundaries, not elimination of GPU waiting. Stage stream times include host
launch gaps; nested stages must not be summed as GPU busy-time percentages.

Target verification remains about 14.83 of 20.73 profiled stream seconds;
packed attention alone is 38.5% of captured CUDA activity. Prefill is about
1.00 s, so batched/chunked prefill was deferred. Next prioritize Target attention
kernels and numerically qualified RMS/RoPE/activation fusion, then targeted
CUDA Graph capture. Ordinary fused operators do not automatically preserve the
reference BF16 rounding sequence. TP2 and page geometry are not the first
bottlenecks demonstrated by this single-GPU workload.

The fixed pool remains 249 pages / 63,744 slots / 9.399 GB. c8/512 peak leases
remain 8,704 slots while peak CUDA allocated memory rises 28.805 → 28.949 GB;
c8/1024 leases rise 11,008 → 11,520 slots as schedules change. c16 stress leases
55 pages / 14,080 slots with 29.365 GB peak CUDA allocation. Shared scratch is
retained while idle for reuse, then `disable_jetspec()` returns allocator
occupancy to zero. No throughput-median regression was observed in the tested
matrix; auxiliary memory and some local bookkeeping costs are real tradeoffs.

Clean production qualification passes all 177 tests with CUDA gates enabled,
plus actual-model isolation, metadata, raw KV and lifecycle gates. Across 315
same-input Draft prediction rows, four cross-layout argmax flips are near ties;
maximum relative RMS error is 0.791% for logits and 0.394% for KV, under the
predeclared empirical 1.5625% envelope. This bound is not a numerical theorem;
strict isolation/history gates provide separate semantic evidence. No exit 139
recurred in these workers; the earlier unreproduced anomaly is not declared fixed.

To reproduce a candidate run, use a clean checkout and the same driver as the
baseline (omit the three optimization flags for clean `7bcb754`):

```bash
python benchmarks/jetspec_phase4.py --repo /path/to/clean-checkout \
  --output candidate.json --concurrencies 1,2,4,8 --outputs 128,512,1024 \
  --lightweight --feature-storage --batched-draft --compare baseline.json
```

The compact evidence includes raw-artifact hashes and paths; full traces remain
outside Git. [The report generator](benchmarks/jetspec_phase4_report.py) pools
completed repeats and checks matching provenance without rerunning GPU work.

See [the qualification record](benchmarks/phase32_qualification.json) for
both latency clocks, per-case acceptance/capacity, source fingerprints,
final-code follow-up and clean-checkout test results. The formal timed runs
precede only a convenience-`generate()` error-handling correction; the
incremental serving/packed execution paths are unchanged, and final code
receives a separate c8/public-API qualification.
The final-code c8 follow-up measures 3.73x and its real-model blocking API
checks pass 9/9. A clean exact-code checkout passes all 130 tests (0 skips),
including GPU packed-attention tests; the new continuous module passes 47/47.
The user's pre-existing uncommitted legacy `runtime.py` remains untouched and
outside this commit: four legacy abnormal-cleanup subtests still fail only
with that retained dirty file, which the new serving runtime never calls.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
