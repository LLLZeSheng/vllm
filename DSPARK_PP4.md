# DSpark Pipeline Parallel（PP4）支持方案

## 结论

本分支以官方 vLLM `v0.28.0`（提交
`2cf0a6915ce544dc493a0990f2ea38d81601128a`）为唯一代码基线，为 DSpark
补齐 Pipeline Parallel 和 1P1D/NIXL 数据通路。它不是简单删除
“DSpark 不支持 PP”的检查，而是修复 Hidden States、Draft token、KV cache、
CUDA Graph warmup 和异构 P/D 拓扑之间的一致性问题。

已验证配置：

```text
P: GLM-5.2 MG39 + DSpark step400, TP1 / PP4
D: GLM-5.2 MG39 + DSpark step400, TP1 / PP1 / DP8
KV transfer: NixlPushConnector
P target graph: FULL_DECODE_ONLY
P DSpark proposal: local eager (target graph remains enabled)
D target/DSpark: CUDA Graph
```

P 和 D 必须使用同一提交。该修改将 NIXL connector metadata 版本升级到 9，
不能与未打补丁的 vLLM peer 混用。

## 原始问题

官方 `v0.28.0` 有两处显式限制：

- `GPUModelRunner` 拒绝 `dspark + pipeline_parallel_size > 1`；
- `load_dspark_model()` 在 PP world size 不为 1 时抛出
  `NotImplementedError`。

仅移除限制仍会出现以下错误：

1. DSpark 依赖的辅助 Hidden States 分布在不同 PP stage，最后一段拿不到完整输入。
2. 最后一个 PP rank 生成的 sampled token 和 Draft token 没有回传，前面 rank 会用旧
   token 进入下一轮 embedding/verification。
3. Draft 的完整层号、embedding 和 lm_head 被错误地按主模型 PP 分片处理。
4. 各 PP rank 的 Draft KV schema 不一致，NIXL 可能注册或发送未计算的 Draft KV。
5. GLM 的 MLA、DSA indexer 与 DSpark Full Attention 组成混合 KV layout；原有
   NIXL 按“每层固定 region 数”切片，在 PP4 和 P/D 异构 TP 下会映射错地址。
6. KV block 的传输长度不一定等于物理 page stride；把二者混用会跨页读写错误数据。
7. CUDA Graph kernel warmup 会在 PP delay line 中留下合成输出，首个真实请求可能消费
   warmup token。

这些问题有的会直接导致初始化或 NIXL 传输失败，有的不会崩溃但会让接受长度明显下降。

## 实现

### PP 内部数据流

```text
PP0             PP1             PP2             PP3
target layers -> target layers -> target layers -> target layers
  |               |               |               |
  +---- hidden/residual/aux slots ---------------->+ DSpark propose
                                                   |
        sampled tokens + draft tokens <------------+
              NCCL broadcast + PP delay line
```

- 每个 PP rank 加载一个完整的 TP1/PP1 Draft replica，使初始化和 KV 规划对称。
- 只有最后一个 PP rank 绑定 Draft attention、执行 proposal 和维护有效 Draft KV。
- 非最后 rank 不声明 Draft KV cache，避免浪费显存及污染 P 到 D 的 KV 传输。
- Draft 配置固定为 PP1；其 layer id 使用主模型全局层数，而不是当前 PP stage 的局部层数。
- PP 模式下 Draft 保留自己的 embedding/lm_head，不共享其他 stage 上的
  `PPMissingLayer`。

### Hidden States 与 token 同步

- `DeepseekV2Model` 为全部 auxiliary layer 建立稳定的全局 slot。
- 每个 stage 写入自己拥有的 slot，并通过 `IntermediateTensors` 继续向后传递。
- PP3 输出完整 auxiliary Hidden States 给 DSpark。
- PP3 将 sampled token 与 Draft token 通过独立 NCCL broadcast 发送给所有前序 rank。
- 前序 rank 按请求 slot generation 过滤已经 abort/reuse 的状态，再更新 Draft token。
- kernel warmup 完成后清空 PP delay line，隔离合成请求与真实请求。

### CUDA Graph

主模型不需要 `--enforce-eager`，可使用：

```bash
--compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

PP4 P 侧的 DSpark proposal 单独禁用 compile/CUDA Graph，避免 Draft 复用 target
input buffer 后将 PP padding sentinel 捕获进 embedding graph。这个隔离只影响小 Draft
forward，主模型 decode 仍走 CUDA Graph。PP1 的 D 侧不受此限制。

### NIXL / 1P1D

- handshake metadata 增加有序 `layer_names` 和 `block_strides`。
- PP producer 与 PP1 consumer 按完整 KV layer 名精确匹配，不再用固定 region 数推导
  offset。
- MLA、Full Attention 和混合 region 分别计算异构 TP source rank。
- 每个 region 只使用其所属 KV group 的 block ids，避免 verifier MLA 和 Draft FA
  互相覆盖。
- 非最后 PP rank 的未计算 Draft KV 不注册、不发送。
- P 比 D 多出内部 speculative suffix 时，传输前把 P block list 裁剪到 D 请求的公开
  prefix，且在 P block 不足时直接报错。
- block transfer length 与物理 page stride 分开计算。

### DP8 profile

DP profile 会在单 worker 上暂时物化全局 token budget。CUTLASS MoE permute scratch
因此按 `max_num_tokens * dp_size` 分配，避免 profile 阶段的
`n_token <= max_num_tokens` 断言失败。

## 文件级改动

| 文件 | 作用 |
|---|---|
| `vllm/config/speculative.py` | Draft 固定使用 PP1 完整拓扑 |
| `vllm/model_executor/models/deepseek_v2.py` | 跨 PP stage 携带 auxiliary Hidden States |
| `vllm/model_executor/models/qwen3_dspark.py` | Draft layer id 使用主模型全局层数 |
| `vllm/v1/worker/gpu/model_runner.py` | DSpark PP 初始化、KV schema、token 同步 |
| `vllm/v1/worker/gpu/pp_utils.py` | sampled/Draft token broadcast 与 delay line |
| `vllm/v1/worker/gpu_worker.py` | warmup 后重置 PP delay line |
| `vllm/v1/worker/gpu/spec_decode/dspark/*` | Draft PP1 配置与 graph 隔离 |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/*` | 精确 region/stride/异构 TP 映射 |
| `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py` | DP profile scratch 容量 |

仓库中的 `patches/dspark-pp4-v0.28.0.patch` 是上述源码修改相对官方
`v0.28.0` 的完整补丁。

## 使用方法

### 直接使用分支

```bash
git clone -b dspark-pp4-v0.28.0 https://github.com/LLLZeSheng/vllm.git
cd vllm
pip install -e .
```

### 给官方 v0.28.0 打补丁

```bash
git clone --branch v0.28.0 https://github.com/vllm-project/vllm.git
cd vllm
git apply --check /path/to/dspark-pp4-v0.28.0.patch
git apply /path/to/dspark-pp4-v0.28.0.patch
pip install -e .
```

### PP4 单服务最小命令

```bash
vllm serve /path/to/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1 \
  --served-model-name glm-5.2-dspark \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 4 \
  --max-model-len 131072 \
  --max-num-batched-tokens 2112 \
  --kv-cache-dtype bfloat16 \
  --attention-backend FLASHINFER_MLA_SPARSE \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config \
  '{"model":"/path/to/dspark/step400","method":"dspark","num_speculative_tokens":8,"draft_sample_method":"greedy","attention_backend":"FLASH_ATTN"}'
```

1P1D 的完整 P/D 命令模板见
`examples/dspark_pp4_1p1d/README.md`。

## 验证记录

源码逐文件 SHA-256 已与三台实际运行节点一致：

- P0/P1：`192.168.0.68`，每实例 TP1/PP4；
- P2/P3：`192.168.0.215`，每实例 TP1/PP4；
- D：`192.168.0.55`，TP1/DP8。

在该代码上已完成：

- PP4 权重加载及服务健康检查；
- `FULL_DECODE_ONLY` target CUDA Graph capture；
- 4P1D NIXL handshake 与混合 KV region 传输；
- 8 个 D engine 均产生非零 Draft/accepted metrics；
- 多轮业务请求无 waiting/error 收尾；
- 修复后业务批次纯 DSpark 平均接受长度约 `3.0–3.6`，恢复到非 PP 对照量级。

接受长度统一按以下公式计算，不加主模型保底 token：

```text
accepted draft tokens / speculative draft iterations
```

## 边界

- 当前实现针对 DeepseekV2/GLM-5.2 模型族的 DSpark 做了端到端验证。
- P/D 必须使用相同提交和 NIXL metadata 版本。
- P 侧 target 支持图模式；P 侧 Draft proposal 当前局部 eager。
- 模型路径、FlashInfer cubin、RDMA/UCX/NIXL 地址仍需按部署环境配置。
