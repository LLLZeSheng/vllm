# DSpark Prefill Context-KV-only 技术报告

日期：2026-10-21。当前 GQA 功能提交：`5b91926f9cc48440fc92622c821e30e7bd3132ee`。

**核心结论：P 侧不必加载完整 DSpark，也能生成并传输 D 侧需要的上下文 KV。**
本次在 GQA PP4 上将 P 侧 DSpark 独占权重从约 **5.553 GiB 降至 0.586 GiB**，
模型加载显存减少约 **5.20 GiB**；0831 的 300 条业务评测未观察到明显接受长度或 TPOT 回退。

本报告以 [GQA-only PR #62923](https://github.com/vllm-project/vllm/pull/62923)
为主线。第 10 节单独记录 MLA PP2、Context-KV-only 及 1M 容量工程试验，
**不表示该扩展已包含在上述 PR 中**。报告仅公开配置和聚合指标，不公开业务请求、
内部机器地址、挂载位置或私有权重文件。

## 1. 结果摘要

所有接受长度均包含每轮主模型输出的一个 token；TPOT 为按输出 token 加权的客户端测量值。

| 路线 | P 侧 DSpark 独占权重，每个 P 实例 | 接受长度，完整 P → Context-only | TPOT，完整 P → Context-only | 结论 |
| --- | ---: | ---: | ---: | --- |
| GQA，P 为 PP4 | 5.553 → **0.585950 GiB** | 4.206129 → **4.183596** | 6.1364 → **5.9190 ms/token** | 正式 PR 路线；两组各 300/300 成功 |
| MLA，P 为 PP2 | 5.918165 → **0.384538 GiB** | 3.741657／3.743017 → **3.782916** | 59.2517／65.4891 → **60.7389 ms/token** | 独立扩展；完整 P 有两次测量，各组 300/300 成功 |

MLA 扩展另外完成两次实际容量请求：

- **1,000,000 输入 + 16 输出**：成功，HTTP 200，收到完整 SSE 结束标记。
- **1,048,560 输入 + 16 输出 = 1,048,576 总上下文**：成功，HTTP 200，完整结束。

这些结果支持“减少 P 侧权重加载且没有观察到明显业务回退”，
**不构成逐 token 严格等价、统计等价、固定比例加速或多条 1M 并发能力的证明**。

## 2. 动机：为什么 P 侧加载完整草稿模型值得优化

### 2.1 P 和 D 的职责不同

PD 分离中，Prefill（P）主要处理已知上下文；Decode（D）负责后续生成。
使用 DSpark 后，D 除了主模型 KV，还需要草稿模型能够使用的上下文 KV。
直接沿用完整草稿模型的构造及加载流程，容易让 P 额外驻留生成阶段才需要的权重。

问题不是“P 不需要 DSpark 信息”，而是：

> **P 必须准备 DSpark Context-KV，但未必需要生成草稿 token 的完整网络。**

### 2.2 PP 末级是容易受影响的显存瓶颈

本次 PP 路径中，DSpark 位于主模型最后一个 pipeline stage，
并不是把草稿层平均分配到所有 stage。
完整 Draft 的 embedding、MLP、输出头等权重集中在末级，会挤压其 KV 预算。

同一个请求必须满足所有 PP stage 的缓存准入。
末级成为瓶颈时，其他 stage 即使还存在空余预算，也不能独立提升整条 pipeline 的容量。
因此，减少末级不必要的 Draft 权重，收益可能不只体现在末级空闲显存，
还可能释放其他 stage 原本受共同 block 数限制而无法使用的缓存预算。

### 2.3 本次目标及非目标

目标：

1. 在模型构造和权重加载阶段就减少 P 侧驻留，而不是加载完整模型后再删除。
2. 保留上下文 KV 的计算、注册、存储及传输契约。
3. D 继续使用完整 DSpark，不改变其采样、验证算法和草稿层数。
4. 以同一批业务请求验证接受长度及 TPOT，而不是只验证服务能启动。

非目标：重新训练模型、修改权重数值、减少主模型层数、取消 Draft KV、
引入有损验证，或者仅通过降低最大长度来制造显存下降。

## 3. 设计依据：Context-KV 路径与 Query 路径可以分离

### 3.1 两类 KV 必须区分

| 名称 | 来源及用途 | 本次处理 |
| --- | --- | --- |
| Target KV | 主模型自己的注意力缓存 | 保留正常主模型执行与传输 |
| Draft Context-KV | 由主模型辅助 hidden states 投影得到，供草稿查询读取 | P 负责生成并传输 |
| Draft Query-tail KV | 草稿查询／生成过程中形成的尾部状态 | D 使用完整 Draft 按既有路径接续 |

“Context-KV-only”不是“只加载一个 FC 就能完成所有工作”，
也不是“把 Draft KV 换成 Target KV”。
两类模型的缓存仍是各自的计算结果。

### 3.2 为什么能省掉 MLP、embedding 和输出头

对于本次 Qwen3 风格 DSpark，已知上下文的 KV 不需要经过完整草稿 decoder 的逐层生成路径。
它先将选定主模型层的辅助 hidden states 合并并投影为共同的上下文表示，
再计算各草稿注意力层的 K/V。

以下为数据依赖示意，具体矩阵存储方向遵循实现：

```text
Selected target hidden states
            |
       Concatenation
            |
       FC projection
            |
        Hidden norm
            |
       +----+----------------+----------------+
       |                     |                |
   Layer 0 K/V           Layer 1 K/V       Layer i K/V
       |                     |                |
  K norm + RoPE          K norm + RoPE     K norm + RoPE
       |                     |                |
   Context cache         Context cache    Context cache

Not traversed here: token embedding -> Q/attention output -> MLP -> LM head
```

令 `z = HiddenNorm(FC(Concat(H_aux)))`，各层主要执行：

```text
K_i = RoPE(KNorm_i(Linear_K_i(z)), positions)
V_i = Linear_V_i(z)                    # 保留配置中的可选 value scale
```

因此 P 要保留 FC、hidden norm、每层 K/V 和 K norm，以及 RoPE 与缓存注册对象。
它不需要为了写入这些上下文 KV 而执行草稿 query、MLP 或词表投影。

**这项分离依赖具体架构的数据流，不能直接套到任意 autoregressive decoder。**
如果另一种草稿模型的下一层上下文依赖完整前层 attention/MLP，必须重新分析。

## 4. 整体方案示意图

### 4.1 优化前后：只改变 P 侧需要构造的草稿部分

```text
BEFORE
  P: full Target + full Draft
       | Target KV + Draft context KV
       v
  KV connector ----------------------> D: full Target + full Draft
                                            |
                                      Draft generation
                                            |
                                      Target verification

AFTER
  P: full Target + context-only Draft
       |             |
       |             +-- FC / norms / K,V / RoPE / cache registration
       |
       | Target KV + Draft context KV
       v
  Same KV connector -----------------> D: full Target + full Draft
                                            |
                                      Query-tail continuation
                                            |
                                      Draft generation
                                            |
                                      Same target verification
```

P 不再产生供 D 使用的有效草稿候选；D 不依赖这些候选来继续生成。
主模型边界状态及请求衔接仍遵循原 PD 协议。

### 4.2 GQA 的单机 1P1D 布局

```text
                    One 8-GPU node

  P instance: TP1 / PP4
  GPU0              GPU1              GPU2              GPU3
  +------------+    +------------+    +------------+    +------------------+
  | PP0        | -> | PP1        | -> | PP2        | -> | PP3              |
  | 22 layers  |    | 20 layers  |    | 20 layers  |    | 16 target layers |
  | no Draft   |    | no Draft   |    | no Draft   |    | partial Draft   |
  +------------+    +------------+    +------------+    | 0.586 GiB       |
                                                      +---------+--------+
                                                                |
                                                 Target + Draft context KV
                                                                |
  D pool: TP1 / DP4 / EP                                         v
  GPU4                 GPU5                 GPU6                 GPU7
  +----------------+   +----------------+   +----------------+   +----------------+
  | complete Draft |   | complete Draft |   | complete Draft |   | complete Draft |
  +----------------+   +----------------+   +----------------+   +----------------+
```

D 的四个 DP worker 结合 EP 工作，图中四格不是四份相互独立的完整主模型。
主模型 PP4 不等于 Draft PP4；Draft 仍采用局部 PP1。

## 5. GQA 实现方式

### 5.1 配置开关与支持边界

新增 `SpeculativeConfig.prefill_context_kv_only`，默认 `false`。
仅在专用 P 侧开启，D 必须保持关闭。
关闭时不新增该选项对应的 graph-hash 因子，继续使用原完整 Draft 加载路径。

当前支持条件：

- `method="dspark"`，Draft 架构为 `Qwen3DSparkModel`。
- `kv_role="kv_producer"`，不接受 consumer 或 producer/consumer 混合角色。
- Model Runner V2；Target 与 Draft 均为 TP1，PCP1、DCP1。
- BF16、无 attention bias、非量化的 Draft 权重。
- Target PP 复用已有 PP-PD 集成路径，本次实测为 PP4。

这里“BF16 Draft 权重”和“FP8 Draft KV”不矛盾：权重与缓存是两种独立精度设置。

### 5.2 选择性构造，而非加载后裁剪

`DSparkContextKVModel` 复用 DFlash 的上下文预计算逻辑，
但构造时只创建必需模块和缓存描述。
它不先分配完整 embedding、MLP、Q/O 投影和 LM head。

层数、辅助 HS 对应的 FC 输入宽度、KV head 数、head dimension、
每层 SWA/Full Attention 均读取配置。
代码不绑定某个 checkpoint 路径，也没有写死“5 层”或“4SWA + 1Full”。

缓存层名继续使用目标层数加 Draft 层索引的注册方式，
与完整模型的 connector 识别和物理缓存布局保持一致。

### 5.3 直接加载 fused K/V 权重

加载器只接受上下文必需的权重集合：

```text
fc.weight
hidden_norm.weight
layers.<i>.self_attn.k_proj.weight
layers.<i>.self_attn.v_proj.weight
layers.<i>.self_attn.k_norm.weight
```

K/V 不再先各存一份再复制到 fused buffer，而是直接写入同一 fused 参数的对应行段。
支持检查点名称中的可选 `model.` 前缀。

必需参数缺失、重复、shape 不一致或 dtype 不一致均报错；
其余只与生成有关的权重不会被加载进部分模型。
这避免用未初始化的随机上下文参数启动服务。

### 5.4 跳过 P 侧草稿查询及补挂完整头

部分模型加载完成后，加载器直接返回，不再补挂主模型 embedding/LM head。
推理执行在上下文 KV 写入后返回无效草稿 ID 占位，不继续执行草稿 query forward。
dummy/profile 与正常执行路径均覆盖该分支。

P 侧部分 Draft 的 CUDA Graph 捕获关闭；这不等于关闭整个 Target 的图执行，
也不影响 D 侧继续使用原 Decode CUDA Graph 配置。

### 5.5 代码分布

| 文件 | 本次职责 |
| --- | --- |
| `vllm/config/speculative.py` | 开关、默认行为、hash 和方法合法性约束 |
| `vllm/model_executor/models/dspark_context_kv.py` | 部分模型、缓存注册、必需权重加载 |
| `vllm/model_executor/models/qwen3_dspark.py` | 构造和加载时选择部分模型 |
| `vllm/v1/worker/gpu/spec_decode/dspark/utils.py` | 跳过完整 embedding/head 共享挂接 |
| `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` | 入口条件检查、部分 Draft 图执行管理 |
| `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | 写完上下文 KV 后跳过 query 生成 |
| `tests/models/test_dspark_mla.py` | 在已有 DSpark 测试文件中增加 GQA 加载与角色测试 |
| `docs/features/speculative_decoding/` | 使用说明及索引入口 |

最后一个测试文件名来自上游已有目录，不表示 GQA-only PR 包含另一类草稿后端。

## 6. 正确性论证及适用边界

### 6.1 为什么 D 侧仍然能够工作

完整路径和部分路径共享相同的主模型辅助特征、FC、norm、K/V 参数、
位置编码、缓存层名和插入规则。
优化删掉的是生成 query 的模块，而不是 D 需要读取的上下文状态。

D 继续加载相同的完整 Draft，按既有路径接续 query tail 并执行验证。
本次同路线对照中 D worker PID 保持不变，
减少了“顺便重启 D 并改变其执行配置”带来的混杂因素。

### 6.2 为什么不宣称逐 token 严格等价

结构上保留所需数据依赖，并不自动意味着不同执行批次、图划分、
BF16 舍入和概率采样轨迹会产生完全相同的输出。
完整 P 与部分 P 的 context GEMM 批次划分也可能不同。

本次证据是：权重子集正确加载、请求对齐、完整业务链路成功、
接受长度及 TPOT 未观察到明显回退。
它不替代跨多种模型、更多随机种子和长时间负载的统计验证。

## 7. 测试设计与指标口径

### 7.1 环境与请求设置

| 项目 | GQA 正式评测设置 |
| --- | --- |
| 主模型 | 内部 GLM5.3 MG43-BNT3，NVFP4 版本 |
| Draft | Alaya 续训练 WTV step800，GQA；权重未修改 |
| GPU | 单机 8 卡；计算能力 10.3，每卡 CUDA 可见容量约 267.69 GiB |
| 软件 | vLLM v0.31.0 原生库 + PP 集成源码；PyTorch 2.13+cu130 |
| PP 集成基点 | `65bad2204be2c848652e6f2c1c888239212ed2f2` |
| KV 传输 | Mooncake CUDA13 0.3.13.post1，RDMA，BLHNC |
| P | TP1/PP4，22/20/20/16 层；batch tokens 32768，max-seqs 8 |
| D | TP1/DP4/EP；batch tokens 128，max-seqs 32 |
| 投机设置 | 8 个 Draft token，probabilistic drafting，block verification，自适应关闭 |
| 缓存 | Target/Draft KV 均 FP8，block size 64，prefix cache 关闭 |
| 执行模式 | 保留原 Target 图设置，D 开启 Decode CUDA Graph；部分 Draft 不捕图 |
| 数据 | 原 0831 固定 800 条中 reservoir 抽取 300 条，seed 20260918 |
| 负载 | 并发 8，最大输出 1024 token，自然 EOS 结束 |

PP4 业务验证使用的是集成环境，不是“未修改官方 v0.31.0 即可完成该部署”的声明。

保留源请求的 temperature、top_p、tools 和 chat-template 参数，没有强制统一温度。
仅为缺失／null 的函数 `parameters` 补空 schema，且两组完全一致。
两组先进行相同长度分位样本预热，不将预热计入正式指标。

一致性检查包括请求体 SHA256、实际输入 token 数、主模型参数尺寸、
P 侧允许的参数集合、D 进程身份及正式计数窗口的干净性。

### 7.2 接受长度

```text
接受长度（包含主模型） = 1 + accepted_draft_tokens / draft_rounds
```

统计来自 D 的所有 DP worker 的累计计数差分，而不是简单平均每个请求的接受长度。
“1”表示一轮验证带来的主模型纠正／bonus token。
spec tokens 开 8 时，一轮理想上限为 9 个输出 token，实际还可能受调度和结束位置限制。

第 i 个位置的无条件保留率：

```text
r_i = 第 i 个 Draft 位置被保留的次数 / draft_rounds
接受长度 = 1 + sum(r_i)
```

无条件保留率不等于“已经到达该位置后”的条件接受率。

### 7.3 TPOT

客户端对输出多于一个 token 的请求计算：

```text
TPOT_j = (流式请求结束时间 - 首个输出到达时间) / (输出 token 数 - 1)
加权 TPOT = sum(TPOT_j * (输出 token 数 - 1)) / sum(输出 token 数 - 1)
```

这是客户端可见的首输出之后的平均生成时延，包含相应流式结束开销，
不是单个 GPU kernel 的耗时；也不是 TTFT，更不是整批总耗时除以总输出 token。

## 8. GQA PP4 测试结果

### 8.1 正式业务结果

| P 模式 | 成功／总数 | 错误 | 接受长度，含主模型 | 加权 TPOT |
| --- | ---: | ---: | ---: | ---: |
| 完整 DSpark | 300/300 | 0 | **4.206129** | **6.1364 ms/token** |
| Context-KV-only，正式公开开关 | 300/300 | 0 | **4.183596** | **5.9190 ms/token** |
| Context-KV-only，较早原型 | 300/300 | 0 | 4.196787 | 6.0182 ms/token |

正式开关相对基线：接受长度减少约 0.022533，即 **0.536%**；
TPOT 观测值减少约 **3.54%**。
结果符合本次 4.0–4.3 的业务验收区间，未观察到明显时延回退。

原型结果单独列出，不替代正式开关结果，也不据单次较优时延宣称固定加速比。
不是重新训练后模型能力提升的实验。

### 8.2 每个 Draft 位置的情况

| 位置 | 完整 P 无条件保留率 | 正式 Context-only 无条件保留率 |
| --- | ---: | ---: |
| 1 | 79.879% | 79.876% |
| 2 | 62.702% | 62.422% |
| 3 | 49.206% | 49.130% |
| 4 | 38.702% | 38.543% |
| 5 | 30.601% | 30.153% |
| 6 | 24.493% | 23.885% |
| 7 | 19.595% | 19.106% |
| 8 | 15.435% | 15.244% |

两条曲线没有出现后段接受率整体坍塌的现象；
但该对照没有足够重复次数用于声明统计等价。

## 9. 显存收益：分别看权重、缓存与整卡占用

### 9.1 实际只加载了多少 DSpark 权重

本次 GQA Draft hidden size 为 6144，使用 5 组辅助特征，
5 个草稿注意力层、16 个 KV head、head dimension 128。
保留参数的 BF16 字节数为：

| 保留项 | 字节数计算 | 实际字节数 |
| --- | --- | ---: |
| FC | `30720 * 6144 * 2` | 377,487,360 |
| 五层 fused K/V | `5 * 2 * 16 * 128 * 6144 * 2` | 251,658,240 |
| hidden norm | `6144 * 2` | 12,288 |
| 五层 K norm | `5 * 128 * 2` | 1,280 |
| **合计** | 上述四项之和 | **629,159,168 = 0.585950 GiB** |

30720 来自 `5 * 6144` 个拼接特征；第一个 K/V 系数 2 表示 K 和 V 各一份，
末尾的系数 2 表示 BF16 每元素两个字节。

模型中是 **8 个参数 tensor**：一个 FC、一个 fused K/V、一个 hidden norm、五个 K norm。
8 个 tensor 不等于 8 层，也不代表 RoPE 等运行 buffer 不再占显存。

原完整 Draft 的独占参数约 5.553 GiB，净减少约 **4.967 GiB**；
另外消除了原实现约 **240 MiB** 的重复 fused-KV 权重 buffer。
因此同口径模型加载 profile 为 **119.58 → 114.38 GiB**，净降约 **5.20 GiB**。

原完整 Draft 名义参数约 7.326 GiB，但约 1.772 GiB 的 LM head 已共享主模型，
不能再次计入“本次节省”。权重文件大小、独占驻留参数和加载 profile 是不同口径。

### 9.2 PP 各级显存及 KV 池

下表来自较早完整／部分模型的成对排空后快照，单位 GiB。
它用于分析物理布局，不冒充正式公开开关业务窗口的连续峰值测量。
正式开关另行确认相同的 8 项权重白名单和 629,159,168 字节总量。

| P stage | 主模型参数 | Draft 参数，前 → 后 | 物理 KV 池，前 → 后 | 排空后 GPU 已用，前 → 后 |
| --- | ---: | ---: | ---: | ---: |
| PP0 | 107.502 | 0 → 0 | 102.643 → 107.833 | 240.355 → 246.393 |
| PP1 | 108.950 | 0 → 0 | 91.978 → 96.629 | 236.525 → 239.850 |
| PP2 | 108.950 | 0 → 0 | 91.978 → 96.629 | 237.562 → 242.402 |
| PP3 | 112.616 | 5.553 → **0.586** | 104.477 → 109.760 | 261.531 → 256.203 |

整条 P 的物理 KV 池从约 **391.077 GiB 增至 410.851 GiB**，增加约 **19.775 GiB**。
末级瓶颈松动后，共同可用 block 数从 **126,632 增至 133,035**。
这解释了为何删掉权重后，一些非末级 GPU 的总占用反而增加：
它们使用了更多缓存，不是优化失效，也不是显存泄漏。

PP3 仍是共同 block 数的瓶颈；其余级约 49.439 GiB 的结构性 KV 预算余量并未完全消失。
该数不包含 utilization 预留，不能把未分配预算伪装成已用缓存。
本次没有为了表面均衡而破坏 DSA indexer-owner 的安全切分边界。

### 9.3 没有省掉 Context-KV

单层 GQA FP8 KV 的逻辑内容量：

```text
2(K,V) * 16(KV heads) * 128(head dimension) * 1(byte) = 4 KiB/token
```

Full Attention 层需要保留完整历史；四个 SWA 层采用 2048 窗口，
但 P 的 chunked prefill 还要容纳尚未结算的 in-flight token。
因此不能把 P 的 SWA 峰值简单按 2048 token 硬截断。

此外，本次多个层的 cache view 共享底层物理 KV pool。
逐层累加 `storage.nbytes` 会重复计算，不能据此报告“Draft 独占 KV 显存”。
应分清缓存逻辑内容、分组准入费用、实际物理池分配和 allocator reserved。

引擎曾估算该 GQA 配置可容纳约 6.17 条 1Mi 长度请求，
**这里只是容量模型估算，本次未对 GQA 完成相应 1M 并发实测**。

## 10. 独立扩展：MLA PP2 与 1M 请求能力

本节记录完整工程工作的扩展验证，不属于当前 GQA-only 上游 PR。
实现及证据独立保留，不能用这里的结果宣传当前 PR 支持所有 MLA 草稿。

### 10.1 部署及部分权重路径

```text
                         One 8-GPU node

   P0: TP1 / PP2                         P1: TP1 / PP2
   GPU0               GPU1               GPU2               GPU3
   +-------------+    +--------------+   +-------------+    +--------------+
   | 38 layers   | -> | 40 layers    |   | 38 layers   | -> | 40 layers    |
   | no Draft    |    | partial MLA  |   | no Draft    |    | partial MLA  |
   +-------------+    | 0.385 GiB    |   +-------------+    | 0.385 GiB    |
                      +-------+------+                      +-------+------+
                              |                                     |
                              +---------- KV transfer --------------+
                                                  |
                                      D: GPU4-7, TP1 / DP4 / EP
                                      Full target + full MLA Draft
```

主模型使用 GLM5.3 NVIDIA NVFP4 版本，Draft 使用 MLA SFT step2400。
Target KV 为 FP8，Draft latent KV 为 BF16，8 个 spec token、probabilistic drafting、
standard verification，自适应和 prefix cache 均关闭，执行模式为 eager。

部分模型只保留 FC、hidden norm、每层 `kv_a_proj_with_mqa` 与 latent norm，
latent／RoPE 维度为 **512／64**。
每个 P 的 Draft 独占参数从 **5.918165 降至 0.384538 GiB**，净省 **5.533628 GiB**；
两个 P 合计约 **0.769076 GiB**。

固定 38/40 分层仍不均衡：两个 stage 的注册模型存储约 194.155／220.183 GiB。
Context-only 减轻了末级 Draft 负担，但没有改变目标层本身的大小差异。

### 10.2 验证中发现并处理的问题

1. **跨层缓存视图不一定连续。** MLA attention 读取改为使用实际 block/token stride，
   不再假设逻辑 `[block, token, latent]` view 连续。
2. **总地址偏移可能超过 32 位。** 在地址乘法前转换为 int64，
   GPU 测试覆盖真实超过 `2^31` 元素的偏移。
3. **MLA Context-KV 写入需要保持原数学顺序。** 保留 norm 的 FP32 归约、
   BF16 转换与缩放顺序，并复用原生 `concat_and_cache_mla` 写入，避免布尔索引造成 host sync。
4. **TP 参考实现的辅助 HS 捕获位置存在集成问题。** 原集成路径曾在延迟 all-reduce 前
   捕获局部 partial；修正汇总后重新建立 TP8 参考。该问题属于验证环境，
   不是本次 GQA-only PR 的代码改动。

修正前的 TP8 接受长度 3.839498 不作为最终参考，避免使用失效基线。

### 10.3 业务结果

| 部署 | 接受长度，含主模型 | 加权 TPOT | 成功／错误 |
| --- | ---: | ---: | ---: |
| TP8，修正辅助 HS 后的参考 | 3.789692 | 87.5216 ms/token | 300／0 |
| PP2 完整 P，首次 | 3.741657 | 59.2517 ms/token | 300／0 |
| PP2 完整 P，独立复测 | 3.743017 | 65.4891 ms/token | 300／0 |
| PP2 Context-only | **3.782916** | **60.7389 ms/token** | 300／0 |

Context-only 接受长度相对 TP8 参考差约 -0.006776（-0.179%）。
完整 PP2 的两次接受长度均比 TP8 低约 1.2%–1.3%，不能宣称完整 PP2 与 TP8 严格零劣化。

Context-only 的 TPOT 比完整 P 首次测量高约 2.51%，比完整 P 复测低约 7.25%。
因此合理结论是“没有明显时延回退”，而不是选择一个基线声称确定加速。
P 的最大长度和 KV 预算也发生调整，这不是只改变一项参数的纯裁剪微基准。

| Draft 位置 | MLA Context-only 无条件保留率 |
| --- | ---: |
| 1 | 77.645% |
| 2 | 58.159% |
| 3 | 43.609% |
| 4 | 32.527% |
| 5 | 24.248% |
| 6 | 18.275% |
| 7 | 13.605% |
| 8 | 10.223% |

### 10.4 1M 容量：实际发请求，不只看缓存预估

| 项目 | 十进制 1M 输入试验 | 1Mi 总上下文试验 |
| --- | ---: | ---: |
| 实际输入 token | **1,000,000** | **1,048,560** |
| 实际输出 token | 16 | 16 |
| 总 token | 1,000,016 | **1,048,576** |
| P 最大上下文 | 1,008,000 | 1,048,576 |
| P batch tokens | 512 | 256 |
| P 每 stage 显式 KV 预算 | 32 GiB | 32.75 GiB |
| TTFT | 265.987 s | 341.073 s |
| 请求总耗时 | 268.452 s | 345.774 s |
| 活动 P 末级最小观测 NVML 空闲 | 1844 MiB | 2068 MiB |
| 请求状态 | HTTP 200，完整 SSE 结束 | HTTP 200，完整 SSE 结束 |

两次试验为重复 token 的合成容量请求，强制输出 16 token，
走完整 P → KV 传输 → D 链路，D 保持原进程和最大上下文设置。
它们不进入 0831 接受长度统计，也不代表 1M 语义质量评测。

必须注意：

- 第二项是 **1,048,576 总上下文**，不是这么长的输入后还能额外生成任意长度。
- 完成的是单条请求；没有证明两个 P 同时承接 1M 或多条 1M 并发。
- 每秒显存采样不是连续瞬时峰值测量，2 GiB 左右观测余量不能泛化为稳定余量保证。
- 原完整 P 在固定 38/40、较高缓存水位下出现过准入不足／长请求 OOM；
  部分权重加上更保守的 prefill 批次才通过本次请求。
  不能把全部收益归因于权重减少，也不能断言所有完整加载配置都不可能处理 1M。

### 10.5 为什么这里的 TPOT 比 GQA 大很多

不能把约 60.7 ms 与约 5.9 ms 的差距归因于 PP2/PP4：
**PP 在 P 侧，两个 D 的布局都为 TP1/DP4/EP。**

两组同时存在以下差异：主模型权重版本、Draft Attention 后端、
eager 与 Decode CUDA Graph、Draft BF16 与 FP8 KV、standard 与 block 验证，
以及 D 的 max-seqs 等配置。
当前 MLA 专用后端不支持 CUDA Graph，是需要优化的执行路径；
没有 profiling 证据，不能把全部差距定量归因于其中任何一项。

接受长度 4.184 与 3.783 只相差约 11%，本身不足以解释约十倍 TPOT 差异。
本报告只在同一路线内部评价 Context-KV-only 的收益。

## 11. 测试覆盖、交付状态与可复现范围

### 11.1 两套测试不能混为一谈

| 测试集合 | 数量 | 覆盖内容 | 对应版本 |
| --- | ---: | --- | --- |
| GQA-only 回归 | **14 通过** | 两种参数名前缀 × 五种加载情形；四种 KV 角色 | 当前 GQA PR，rebase 后重跑 |
| 完整双架构验证 | **27 通过** | 5 项选择性加载、4 项角色、6 项 MLA 写入、12 项 stride/地址读取 | 保留的扩展候选版本 |

GQA 加载测试覆盖：两层 fused K/V 行顺序、完整加载，以及缺失、重复、shape、dtype 错误。
扩展 GPU 测试还覆盖两种 RoPE 布局、contiguous/BLHNC/BNLC、padded slot、
causal/windowed attention 和真实大地址偏移。
27 项不是当前 GQA-only PR 自身的测试数量，也不能与 14 简单相加作为一套回归。

本地还通过相关 Ruff、格式、Markdown、mypy、配置字段、导入、SPDX、
GPU API／同步约束及 Buildkite 测试挂接检查。
新 GQA 测试位于现有 basic-model CI 已选择的文件中。
私有模型和业务数据无法直接进入上游 CI，业务验收是本地证据，不冒充上游 CI 覆盖。

### 11.2 版本与 rebase

- GQA PR：[vllm-project/vllm#62923](https://github.com/vllm-project/vllm/pull/62923)。
- GQA 功能提交：`5b91926f9cc48440fc92622c821e30e7bd3132ee`。
- Rebase 基点：`817fa3518cd4bcc94b31e044c534c002e701c801`。
- Rebase 只手工处理文档索引插入位置，功能补丁保持不变；14 项 CPU 测试及相关检查重跑通过。
- 业务数据来自原始集成环境验收，**不是 rebase 后重新进行的 300 条完整压测**。
- 报告生成时，PR 为 Open、可合并无冲突，但仍需 review；自动 `pre-run-check` 显示失败，
  因此**不能将本地检查通过写成上游 CI 全绿**。
  [该次 Actions 日志](https://github.com/vllm-project/vllm/actions/runs/39626775916)
  显示失败原因是执行资格检查：PR 缺少指定标签且作者已合并 PR 数为 0，未达到 4 个的门槛；
  不是代码测试断言失败，后续 pre-commit job 被跳过。本次报告提交不修改标签或 CI 策略。

### 11.3 启用方式

仅在具有对应功能代码的专用 P 上，将以下字段合并到现有配置：

```json
{
  "method": "dspark",
  "model": "<same-draft-checkpoint-as-decode>",
  "num_speculative_tokens": 8,
  "draft_tensor_parallel_size": 1,
  "prefill_context_kv_only": true
}
```

P 的 KV 角色为 `kv_producer`；D 使用同一检查点并将此选项省略或设为 `false`。
其余采样、验证、缓存及 connector 参数保持同路线一致。
这不是完整可复制的 PP4 部署脚本；PP 集成、router 和传输环境仍需单独准备。
不要将只支持 prefill 的 P 当作独立多 token 生成服务。

本功能无需修改 safetensors，也无需在权重目录附带自定义运行时代码；
**前提是运行的 vLLM 已包含相应功能及部署依赖**。

### 11.4 回归入口及公开证据

配置好源码与测试环境后，可使用现有测试入口：

```bash
.venv/bin/python -m pytest tests/models/test_dspark_mla.py -k context_only -q
```

实际本地隔离 wrapper 导入候选模块，并用 `--noconftest --import-mode=importlib`
运行选定单测，避免修改已驻留服务的源码。
报告同目录的 [validation-summary.json](validation-summary.json) 保存了
本报告引用的聚合数值与每位置结果，不含原始请求。

需要完整复现业务测量时，还必须具备相同检查点、请求体、分词器／模板、
采样参数、预热方式、缓存布局、软件集成和计数窗口；仅有本报告并不足以复现私有数据集。

## 12. 局限与后续方向

已完成：GQA 选择性加载、PP4 PD 业务验收、P 侧权重和缓存物理核算；
独立扩展完成 MLA PP2 及两次实际长上下文请求。

仍有限制：

1. 当前 PR 仅支持指定 GQA 架构和并行／精度组合，不能宣称任意 DSpark 通用。
2. D 仍加载完整模型，Context-only 没有减少 D 的模型权重或主模型计算。
3. PP 显存并未完全均衡，固定安全分层、共同 block 数和临时工作区仍制约容量。
4. 扩展 MLA 的四层 SWA mask 生效，但 allocator 仍按 Full MLA Cache 保留；
   不能据窗口大小直接宣称已回收对应历史缓存。
5. 没有多条 1M 并发、长输出和长时间压力测试结论，也没有逐算子性能归因。

建议后续分别推进：更多支持配置的正确性回归、MLA Decode 图执行／后端性能分析、
保持 DSA 依赖的 PP 安全平衡、MLA SWA 物理缓存管理，以及长上下文并发压力验证。
这些是后续工作，不作为本次交付已经完成的能力。

## 13. 代码与相关工作

- [GQA-only 功能提交](https://github.com/LLLZeSheng/vllm/commit/5b91926f9cc48440fc92622c821e30e7bd3132ee)。
- [GQA 部分模型实现](https://github.com/LLLZeSheng/vllm/blob/5b91926f9cc48440fc92622c821e30e7bd3132ee/vllm/model_executor/models/dspark_context_kv.py)。
- [功能使用说明](https://github.com/LLLZeSheng/vllm/blob/5b91926f9cc48440fc92622c821e30e7bd3132ee/docs/features/speculative_decoding/context_kv_only.md)。
- [PP-PD 集成 PR #56957](https://github.com/vllm-project/vllm/pull/56957)：说明 PP 业务验证所使用的集成背景。
- [保留的完整扩展候选提交](https://github.com/LLLZeSheng/vllm/commit/ef73a6ae64a2ffb9daf93d2ea20e20b8c645b498)：不属于当前 GQA-only PR。
- [#55654](https://github.com/vllm-project/vllm/pull/55654) 和 [#60076](https://github.com/vllm-project/vllm/pull/60076)：主要优化 KV 插入，不等同于构造前减少 P 侧权重。
- [#60320](https://github.com/vllm-project/vllm/pull/60320)：裁剪 trailing-window context 计算，与本次权重选择性加载的切入点不同。

本工作使用 OpenAI Codex 辅助开发、验证与整理；实现由提交人审查并确认运行相关代码。
所有吞吐／时延和容量结论均限定于上述测量环境及协议。
