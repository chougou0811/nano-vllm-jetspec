# Phase 5：官方经验迁移与 Target 执行优化

本轮优化 Target tree verification 的 prefix 地址解析、启动开销和 RoPE/KV scatter；
不改变 Draft 权重、tree budget、serving admission 或原始 benchmark workload。
性能数据以独立 clean worker 的真实 median 为准，没有因结果改预算或 workload。
选定候选为 `bc0df2498a229b3c5a2904088a867dd78d7cefb4`（graph + reference-order fused RoPE）；
production SHA256 为 `dc54bd99707e7b46d4f4ac0a07f5cb04b5434c2e942033798da8cd69ae4de6ff`。

## 为什么不能直接追官方 9.64×

官方数据有三个不同口径，不能混用：

| 官方数字 | 条件与分母 | 与本项目的区别 |
|---|---|---|
| MATH-500 9.64× | Qwen3-8B、H100、greedy、budget256、离线任务评估 | 不是相对 pristine nano-vLLM 的 matched serving 加速比 |
| MATH-500 1150 tok/s | B200、batch1、graphed Draft/verify、fused GEMMs、no-gather、session、warm steady-state | 我们使用5090、ragged multi-request、动态到达、mixed output caps |
| verify-only 7.31×/7.55× | Target GPU 时间，排除 Draft、prefill 和一次性 graph capture | 不能当作端到端 throughput speedup |

官方 reference benchmark 的 AR 分母是 raw HuggingFace KV-cache greedy；
源码分别输出排除 prefill/setup 的 `speedup` 与完整 wall-time 的 `e2e_speedup`。
优化引擎的 wall-time driver 则串行处理任务 prompt，并以自身 engine AR 作分母。
我们的 pristine FlashAttention nano-vLLM 分母更强，真实 arrival/prefill/delivery 成本也进入计时。
因此性能差距同时包含硬件、任务 acceptance、预算、batch、分母和执行实现差异。

Primary sources（仓库固定在 `2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f`）：

- [官方 README：配置和 B200 engine results](https://github.com/hao-ai-lab/JetSpec/blob/2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f/README.md)
- [reference benchmark：HF AR 与 decode/e2e 口径](https://github.com/hao-ai-lab/JetSpec/blob/2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f/bench/reference/benchmark.py)
- [engine wall-time benchmark](https://github.com/hao-ai-lab/JetSpec/blob/2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f/bench/engine/tps_walltime.py)
- [verify-only 对比定义](https://github.com/hao-ai-lab/JetSpec/blob/2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f/bench/profiling/compare_engine_with_vllm_integration.py)
- [官方论文](https://arxiv.org/abs/2606.18394)

## 实际迁移的三条路径

`tree_prefix.py` 对完整 prefix tiles 采用 page-contiguous 地址解析：每 tile 读取一次页表，
跳过不必要的 tree-slot/ancestor metadata；prefix tail 与 tree 仍用原可见性和地址映射。
保留原 key tile 顺序、FP32 QK/PV reduction 与 online softmax，避免换成不同算术的 attention。
`paged_backend.py` 的 auto dispatch 只在已测的 prefix/并发条件启用，短 prefix 继续 reference。

`tree_fusion.py` 将 HF-order RoPE 与 provisional KV scatter 合为一个 Triton kernel。
输入仍是原始 split Q/K/V GEMM 和 reference Q/K RMSNorm 的结果。
每个 BF16 multiply 保留原有舍入边界，再执行 add/subtract；不是 FP32 融合后仅末尾 cast。
关闭浮点 FMA fusion；每个 KV head 只有一个 writer，GQA 其余 head 只写 Q。
写入 transaction 已分配的独立 scratch slots，不写 canonical prefix。
Serving 用 `rope_scatter_prevalidated` launch-only 入口，省去中间张量和多个 eager 启动。

`target_graph.py` 借鉴官方 persistent staging + CUDA Graph replay，而非整段照搬。
参考：[官方 GraphedVerify](https://github.com/hao-ai-lab/JetSpec/blob/2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f/jetspec/inference_engine/graph_capture.py)。
我们 capture 现有 packed Target forward，包括 lm_head；不重新定义 attention 算术。
GEMM 保留真实 `Q=sum(tree_nodes)`，不 padding tree rows，避免人为改变执行 shape。
Graph key 包含精确 Q、请求数、ragged mask 元素数及 reference/prefix attention dispatch。
每轮更新 token、position、prefix length、request/local row、node counts、ancestor bias、
query offsets、block tables 和 scratch slot mapping；请求 ID 和地址值不会作为静态语义缓存。
Prefix length 是 device 值，attention 循环不冻结 capture 时的历史长度。
Block table 宽度固定为 `ceil(max_model_len/page_size)`；4096/256 对应16列。
最多16个 graph entries，超出容量的新 shape 回退原 eager 路径，不无限扩展或偷偷调预算。
所有 graphs 使用一个 persistent capture stream 和共享 graph memory pool。
执行顺序仍是 batched Draft → packed Target verify → 各请求 accept/commit → exactly-once delivery；
本轮没有新增 Draft CUDA Graph，也没有把 prefill 或 canonical KV commit 放进 Target graph。

## Ownership 与 serving 接口

Graph cache 属于 runner；canonical KV 属于 request，Tree Scratch 仍属于 batch/runner。
Capture/warmup/replay 只能写本轮 scratch，acceptance 和 accepted-only physical commit 在 graph 外。
输出 logits/taps 是 borrowed buffer：本轮串行 acceptance/feature append 必须消费完再下一次 verify。
需要跨轮保存的诊断必须 clone，不能把 borrowed view 当作永久 reference。
Scratch retirement event 覆盖消费和 commit；异常 cleanup 先 fence capture side stream。
Graph 输入、模型/KV 地址保持稳定；不支持并发 graph 执行或任意跨 stream 保留输出。
Idle disable/reenable 可保留 warm graph；改变 kernel policy 会关闭旧 graph，runtime close 负责释放。

Graph/RoPE policy 默认仍是 `eager/reference`，ordinary AR 不变；主动开启：

```python
# LLM 普通 runner 仍以 TP=1、enforce_eager=True 初始化。
llm.configure_jetspec(
    draft_path,
    optimization="serving",
    attention_backend="sdpa",
    enable_chunked_prefill=False,
    target_execution="cuda_graph",
    target_kernels="fused_rope",
)
```

## 保留的 negative result

Reference-order RMSNorm prototype 保留为显式实验模块，不接入 serving API。
Strided/native-BF16 witness 未通过既定 local numerical bound：
相同 FP32 square/mean/rsqrt 表达式仍可能因 reduction order 和 BF16 舍入边界产生差异。
没有为留下优化而放宽门限；production 只提供 `reference/fused_rope`。
官方 [fused GEMM stack](https://github.com/hao-ai-lab/JetSpec/blob/2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f/jetspec/inference_engine/compiled_verify_stack.py)
会合并 Q/K/V 与 gate/up 权重；低成本迁移原型的 same-state sanity 未通过，未入选。
其 final hidden/logits relative RMS 为0.01796/0.01868，高于既定0.015625；不因速度动机放宽 contract。
这说明不同 GEMM shape 的 BF16 误差需要实证处理，并非“矩阵拼接后一定错误”。
官方 attention 的 BF16 probability Tensor Core 路径也没有直接替换我们的 FP32-P tree backend。
本轮保留 Phase3.1/3.2 numerical/isolation contract，不要求跨不同 layout 的 token bitwise 等价。

## 真正 system-to-system benchmark

固定 RTX5090、Qwen3-8B BF16、TP1、greedy、max_model_len4096、gpu_memory_utilization0.8。
沿用 `phase5_workload_manifest.json` 的全部 token IDs、mixed budgets/caps 与 two-wave arrivals；
128/512 是 output-cap scale，不代表每个请求都输出相同长度。
每组 c1/c4/c8 × output128/512：独立 case cache reset（计时外）→warmup→至少3次正式测试。
Case 内每次冷启动 page allocator，模型/Draft 与 warm graph 常驻；16-entry 上限不变。
记录 formal capture/fallback：发生的冷 capture 成本计入结果，不能称为无 capture 的 steady-state。
Baseline 是 pristine upstream `df99418f7d6ca676550f4372cdc6e1521ce8c33d`，实际走两种 FlashAttention API。
项目起点 `bb823b3` 已移除 greedy，故采用其最近之前原生 greedy ancestor，不修改 upstream。
旧 JetSpec 臂是冻结 Phase4 `b388330d45bf42adf7e0112319c031587d9828cb`，不是后续中间版本。
因此本轮相对旧 JetSpec 的增量包括 qualified prefix attention、RoPE/scatter 和 Target Graph。
Upstream eager 与 JetSpec Target graph 是明确的系统差异；此比较不是 backend-matched ablation。
吞吐 = 实际 delivered output tokens / synchronized wall time；speedup = 两臂吞吐 median 之比。
TTFT/E2E 分 offered 与 submitted clock；delivery TPOT = `(last-first)/(tokens-1)`，不是内部逐token ITL。
同时保留 gap p95/max、effective tokens/verify、native KV pool、allocator cleanup 和 PyTorch peak memory。

| case | pristine upstream FA tok/s | 原 JetSpec tok/s | 优化 JetSpec tok/s | vs upstream | vs 原 JetSpec |
|---|---:|---:|---:|---:|---:|
| c1/output128 | 47.72 | 144.18 | 207.33 | 4.34× | +43.80% |
| c1/output512 | 48.56 | 159.23 | 226.06 | 4.66× | +41.97% |
| c4/output128 | 150.57 | 276.29 | 292.73 | 1.94× | +5.95% |
| c4/output512 | 160.15 | 348.37 | 379.63 | 2.37× | +8.97% |
| c8/output128 | 228.51 | 300.87 | 316.92 | 1.39× | +5.33% |
| c8/output512（预注册简历 case） | 244.41 | 385.09 | 413.36 | 1.69× | +7.34% |

原始数据、source pins 和 median 由 `jetspec_official_port_benchmark.py` 生成；不改历史 JSON/README。
简历只能引用新鲜 matched median；历史7.27×属于 fork SDPA AR 对比，不能称“相比原版 nano-vLLM”。

### c8/output512 的真实取舍

每次测试16个请求、共5632 output tokens，沿用两个 arrival waves 和 mixed caps。
三次正式 tok/s 全部保留：upstream `[244.414, 244.812, 234.902]`；
Phase4 `[384.958, 385.089, 385.763]`；优化版 `[413.356, 413.699, 412.907]`。
没有删掉 upstream 较慢的一次，也没有挑优化版最好的一次。

| 指标 | upstream FA | 冻结 Phase4 | 优化 JetSpec |
|---|---:|---:|---:|
| Offered TTFT p50（s） | 0.201 | 1.143 | 1.085 |
| Offered E2E p50（s） | 11.909 | 10.299 | 9.777 |
| Delivery TPOT p50（ms/token） | 27.048 | 14.687 | 13.908 |
| Inter-delivery gap p95（ms） | 22.502 | 232.975 | 224.439 |
| Inter-delivery gap worst max（ms） | 11849.703 | 413.317 | 409.219 |
| Peak PyTorch allocated（GiB） | 24.477 | 27.530 | 28.688 |
| Peak PyTorch reserved（GiB） | 24.730 | 28.869 | 30.625 |
| Peak leased KV pages | 38 | 53 | 53 |

Latency 表是各正式 sample 的 per-request 统计再取 median；gap max 是全部三次的最坏值。
吞吐/TPOT/E2E 改善不等于所有体验指标改善：对原版，JetSpec TTFT 和 gap p95 均更差。
原版 scheduler 优先 prefill waiting，admission 的 `num_seqs` 计本轮而不是 running 总数，
decode 又从 running 队头选取；它能很早交付新请求首 token，随后出现长 decode 等待。
JetSpec 使用严格 active admission、round-robin 和多 token block delivery，因此首 token 排队及
burst gap 行为不同。这是保留两边自然 serving policy 的实际系统对比，不是统一调度器实验。
不能声称“TTFT 更好”或“所有 latency 都更好”。

优化版相对 Phase4 allocated 多约1.16GiB，reserved 多约1.76GiB；native KV pool 仍为
249页 × 256 slots，allocator 最终全部归零。每次 packed verify 有效输出39 tokens，
每个 verified request 为7.468 tokens，与 Phase4 相同；本轮没有提高 acceptance。
18次正式 sample 共1497次 graph replay，0 capture、0 eager fallback。
这是每 case warmup 后的 steady-state 收益：c8/output512 warmup 18.178s，旧版14.627s，
不能把 warmup 差直接全部当成独立 graph capture cost，也不能外推为更快冷启动。
未知新 shape 或16-entry cache满时仍执行 eager，不能保证任意线上 shape 复现同样收益。
Upstream 自己也有 CUDA Graph，本轮依既定配置保持 `enforce_eager=True`；
1.69× 不代表相对原版开启自身 CUDA Graph 后的加速比。

### 一次有目的的 bounded profile

只追加选定候选的 c8/output512 晚期4轮 CPU/CUDA profile，没有再跑全面测试。
实际 Q 为 `[376,360,360,360]`，其中一轮有动态1024-token新请求 prefill。
选定 `bc0df24` 的 individual CPU kernel launch为6670，另有4次 `cudaGraphLaunch`；
GPU仍执行13482 kernels，其中 Target verify为6856 kernels。
历史 qualified prefix/eager `f1175bb` 在相同Q/逻辑prefix窗口记录16358个 individual CPU launches，
GPU16358 kernels（Target9732）。这是已有 profile 的诊断对照，不是新的 matched profiler A/B；
不能将 kernel 数下降全说成Graph的功劳：RoPE/scatter fusion减少GPU kernels，Graph主要减少host launches。
两份窗口 synchronization count 都是142（140 stream + 1 event + 1 device），
本轮没有实证消除这些 host sync；不要在简历写“同步下降59%”。Profiler本身也可能引入同步。
新 Target summed GPU busy time为258.78ms：GEMM合计169.16ms、tree attention73.66ms、
其余15.96ms。该总和不是 wall latency；graphed GEMM不能从已消失的逐层Python range
准确拆成MLP/QKV。当前优先瓶颈仍是 Target GEMM及 FP32 tree attention，随后是Draft/metadata/同步，
不是 accepted KV copy。没有为凑模块继续实现未证明收益的更大优化。

完整可审查证据：[phase5_official_port_benchmark.json](../benchmarks/phase5_official_port_benchmark.json)。
该 JSON 包含三臂全部 warmup/正式记录、每组 median、FlashAttention 实际调用证明、
已有同状态诊断、未入选 GEMM witness及一次bounded profile；没有新增全面资格测试。

### 简历写法

第二项目建议命名“Ragged Tree Target 执行栈优化”，不是把全部系统加速称为纯 attention 算子收益。

> 基于 profiling 实现页连续 Tree Attention 地址优化、Triton RoPE/KV Scatter融合与
> 精确 shape CUDA Graph执行缓存；在RTX5090/Qwen3-8B BF16、c8/output512混合负载下，
> 吞吐较冻结JetSpec提升7.34%至413 tok/s（3次正式median）。

整体项目的系统结果可另写：同条件达到原版 FlashAttention nano-vLLM **eager baseline** 的1.69×吞吐。
该1.69×同时包含 JetSpec speculative decoding、已有 serving 优化和本轮 Target执行优化；
不能把它全部归因本轮 kernel，也不能与历史SDPA的7.27×混用。
