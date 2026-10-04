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
AR-token equivalence for all serving schedules. Do not advertise this path as
fully lossless multi-request decoding until that gate is resolved.

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


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
