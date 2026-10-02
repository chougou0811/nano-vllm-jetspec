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
The current attention numerical path remains c1-only. Packed ragged kernels,
scheduler admission, multi-request cancellation, prefix sharing, CUDA graphs
and page-size comparisons require separate qualification.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
