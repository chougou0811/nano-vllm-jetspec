# Phase 5: Target verification Tree Attention

本轮只优化 **Target verification**：Draft、prefill、tree policy、admission、accept/commit 和 shared scratch ownership 不变。正式 serving 测试关闭 Chunked Prefill，保持 SDPA prefill/Draft、BF16、TP1、eager、greedy。

## 实际接入的算子

`paged_backend.packed_tree_attention` 默认 `backend="auto"`；`backend="reference"` 可显式复现冻结的旧 kernel。已测 SM120 / Qwen3 BF16 geometry（32 Q heads、8 KV heads、D128、page256）自动选择 prefix specialization；c1 prefix <256、multi-request prefix 全部 <64，以及未验证的其他 geometry/device 仍走 reference。这些条件不改变 tree budget 或调度。

独立 checked API：

```python
from nanovllm.speculative.jetspec.tree_prefix import packed_tree_attention_prefix_split_exact

out = packed_tree_attention_prefix_split_exact(q, k_pages, v_pages, metadata, scale, 4)
pre_round = packed_tree_attention_prefix_split_exact(
    q, k_pages, v_pages, metadata, scale, 4, output_dtype=torch.float32)
```

`q` 为 `[sum(T_i), Hq, D]`，K/V 为 `[pages, page_size, Hkv, D]`，支持实际 pool 的非连续 layer/page strides。每个 CTA 仍对应一个 `(packed query, head)`；不是 Tensor Core/GQA reuse 算子。

完整 prefix TILE64 一定位于单个 page256 内：一次 scalar page-table load 即可得到整个 tile 的 page/offset，不再逐 key 做 prefix/tree slot selection。最后不足64的 prefix tail 从 `floor(P/64)*64` 开始，继续使用原始 mixed prefix/tree addressing，因此没有改变 reduction 分组。

两段循环保留原始 FP32 QK、`scores + bias`、visibility、exp、PV、running-max/sum 更新表达式。Tree 部分仍使用每个请求独立的 scratch slots 和 ancestor bias；仅按 packed 索引做 flat causal attention 是错误的。算子只读 K/V，不负责物理 commit。

`split_exact` 指源码算术表达式一致，不是跨环境 bitwise 保证。独立入口负责完整验证；serving 在 dispatcher 已验证 geometry 后复用同一 JIT launcher，避免再次验证，并沿用 `empty_like` 输出创建。device-capability 查询缓存；选择依据来自 host metadata，不增加 `.item()` / `.tolist()` 或 CUDA 同步。

## 资格与实测

生产代码实际测量快照：`f1175bb`；内部基线为 `6f65f6e` 的 Final JetSpec SDPA 路径，不是 pristine upstream AR。源码、模型路径、环境、KV geometry、manifest 和已执行入口均绑定到报告。

固定 gates：原生 BF16/FP32 operands 的 pre-store FP32 output 对独立 CPU FP64 parent-chain oracle，scaled-max 和 relative-RMS 均为 `1e-4`；整网经验 BF16 gate 为 `2^-6`。不要求跨 execution layout 的 AR token bitwise 等价。

边界 P0/1/63/64/65/255/256/257/1024/2048、随机非连续 pages/slots、ragged trees、有限跨请求/分支扰动均通过。真实 Qwen3 两轮同状态重放的全层 scratch KV、taps、logits 恰好 byte-exact；动态 arrival、EOS、cancel、max_tokens、allocator pressure、preemption/chunked recompute 和 cleanup 通过。端到端18组正式样本的请求输出也与基线相同；这是观测结果，不升级为通用 bitwise contract。

冻结原 final workload：mixed prompts 128/1024/2048、mixed tree budgets 63/31/47、mixed output caps、两波 arrival。每个 case 1 warmup +3正式样本，保留所有样本取 median：

| concurrency / output scale | 优化前 tok/s | Phase 5 tok/s | 变化 |
|---|---:|---:|---:|
| c1 /128 | 141.65 | 139.39 | -1.60% |
| c1 /512 | 158.69 | 156.95 | -1.10% |
| c4 /128 | 272.36 | 277.89 | +2.03% |
| c4 /512 | 344.78 | 351.77 | +2.03% |
| c8 /128 | 297.92 | 303.08 | +1.73% |
| c8 /512 | 382.25 | 389.91 | +2.00% |

这是小幅收益，**不是普遍加速**；c1 的下降原样保留，3次测量不支持统计显著性声明。c8/512 是预先指定代表 case，实际含16个请求、5632输出 tokens，不是16个请求都输出512。

c8/512 offered TTFT 1.155→1.151s、delivery TPOT 14.768→14.436ms、offered E2E 10.376→10.155s；各项为三次 request-p50 的 median。delivery gap p95 233.879→230.422ms、所有正式样本 gap max 414.645→411.049ms。emitted/request-verify 保持7.468；pool249 pages×256、peak leased53 pages、peak PyTorch allocated27.530GiB均不变；cleanup全部归零。

## 拒绝的候选与下一步

- Tensor Core QK + FP32 probability / TF32x3 PV：synthetic micro 长 prefix 约5×，独立 FP64 gate通过；但真实整网 lm_head relative-RMS 0.016753 超过0.015625，KV/taps/hidden gate也失败，**没有接入 default serving**。
- 简化 prefix math 的首版：micro约1.3×，整网仍超出固定阈值，拒绝。保留 bias/visibility 算术后只剩约1.05–1.07× micro收益，资格通过。
- broadcast GQA reuse、register cap80、register cap96：负收益或不足以补偿 c1 回归，不启用。首版 dispatcher 的 c1/512 -3.45%也保留，最终轻量入口为-1.10%，没有丢弃不利样本。

c8 四轮 profile 的 Tree Attention kernel busy time 78.321→73.992ms；runtime/driver launch count均16358，sync count均142。本轮没有消除 launches/sync，也没有做 CUDA Graph。当前 Target GPU busy 的最大部分仍为 MLP GEMM（约116.8ms），大量 reference RMSNorm/RoPE/slot 小算子和 eager launch 开销值得继续研究 fusion / bucketed Target CUDA Graph；不要把 GPU busy 占比当作端到端收益预测。

Nsight Compute 2025.1.1 实测报 `ERR_NVGPUCTRPERM`，没有修改宿主权限；没有硬件 occupancy/bandwidth 数据。报告使用真实 CUDA traces 和编译器 n_regs/n_spills，明确区分两者。旧 c8 profile 没有 `prefix_lengths_before_step` 字段；对两份 profile 的共同 shape/delivery 字段核验一致，不补造旧 prefix 数据。

## 复现入口

使用项目安装环境，并保证官方 `jetspec` 可导入。`benchmarks/phase5_workload_manifest.json` 与原 final manifest **字节完全相同**。以下均为已有 driver，不需要修改 production：

```bash
python benchmarks/jetspec_tree_kernel_benchmark.py --help
python benchmarks/jetspec_tree_kernel_qualification.py --help
python benchmarks/jetspec_tree_serving_benchmark.py --help
python benchmarks/jetspec_tree_serving_profile.py --help
python benchmarks/jetspec_phase5_report.py --help
```

micro显式指定 `nanovllm.speculative.jetspec.tree_prefix:packed_tree_attention_prefix_split_exact --num-warps 4`；qualification指定 `tree_prefix:packed_tree_attention_prefix_split_exact --query-tiles 1 --num-warps 4`。Serving用两个独立 clean checkout，显式 pin HEAD/production SHA，Final入口 witness为 `packed_tree_attention_prefix_prevalidated`。最后publisher必须绑定同一 candidate source 的micro/qualification/serving，不接受pre-dispatch资格作为最终资格。

完整数字、原始3次样本、失败 witness、source fingerprints及 trace SHA在 `benchmarks/phase5_qualification.json`。README、历史 final benchmark JSON、用户未提交的legacy `runtime.py`均未修改；该dirty runtime原有4个异常清理子测试错误仍保留，clean serving快照399项单测通过（7项默认跳过）。
