# GLM-5.3 MLA DSpark deployment integration

This fork integration adds a GLM MLA drafter, per-layer windowed latent caches,
prefill context-KV-only (PDCO), and MRV2 target/draft CUDA Graph execution to the
existing v0.31 PP/PD deployment branch. It does not require Python runtime files
inside the checkpoint: the draft directory contains its configuration and
ordinary safetensors weights, and the implementation lives in the vLLM package.

The PR targets `LLLZeSheng/vllm:deploy/pp2-dspark-pdco-v031`. It consolidates the
previous MLA extension and the new validated graph path. It is not a second
upstream submission competing with [GLM MLA PR #62410](https://github.com/vllm-project/vllm/pull/62410).
Windowed attention reuses [PR #63795](https://github.com/vllm-project/vllm/pull/63795),
commit `3de196f40700bf732ab260bf74883ad13c21f02a`, in the existing
`vllm/v1/attention/ops/triton_decode_attention.py` module. No duplicate custom
attention kernel is added.

## Implementation and scope

```text
P: target PP2 (40/38)
   target auxiliary layers -> GLM fusion/RMS -> latent512 + rotary64
   target KV + draft context KV -- Mooncake --> D

D: target TP1 / DP4 / EP4
   target verification graph (K+1 rows/request)
   dense MLA draft graph (K rows/request) -> existing Markov sampler
   existing standard or block verifier -> accepted prefix + target token
```

- Register `GlmDSparkMLAForCausalLM` and its configuration/weight mappings.
- Preserve the trained RMS normalization, rotary convention, and Markov head.
- Use native `concat_and_cache_mla` for strided latent-cache writes.
- Register `SlidingWindowMLASpec` for each causal windowed layer. The query
  adapter uses each query's own position, masks padded/empty requests, and
  never reads released prefix pages outside its window.
- Advertise uniform-batch graph support. The draft previously advertised
  `NEVER`, disabling the target graph through MRV2's global graph resolver.
- Give the dense TP1/DCP1 MLA draft an independent request-multiple capture
  grid and local dispatch. Other drafters retain the existing dispatch;
  target DP/EP collectives and synchronization are unchanged.
- P loads the context-KV subset; D loads the complete drafter. Target PP
  auxiliary-state carry and its reduction preserve the trained fusion input.

Validated MLA dimensions are q-rank 2048, latent-rank 512, no-RoPE head 192,
RoPE head 64, and value head 256. The measured checkpoint has 64 query heads,
five causal SWA8192 layers, auxiliary target layers `[2, 21, 39, 57, 76]`, and
unchanged BF16 weights. Draft execution PP1/DCP1 is the validated mode; the
target can use PP2. Noncausal full attention is supported; noncausal SWA is
explicitly rejected because its per-query window differs from the reused
kernel's sequence-end window. Do not extrapolate this to generic DCP support.

## Validated environment and results

The evidence uses vLLM 0.31.0 native libraries with the integration source,
Python 3.12, Torch 2.13+cu130, FlashInfer 0.7.0.post1, and MRV2. The device reports
`NVIDIA_H800`, CUDA SM103a, approximately 267.69 GiB per GPU. These results are
not certification of an unmodified release or current upstream main.

The validation runner also used a task-local FlashInfer artifact-download retry
shim and read-only worker RPC diagnostics. The shim preserves cubin hash
verification; its hidden-state extraction cache workaround does not apply to
MG46. Neither is a checkpoint runtime override or part of the model arithmetic.
Normal FlashInfer kernel artifacts must be available in a standalone install.

The [comparison report](COMPARISON.md) records the MG43, context196608,
0831 fixed300 evaluation, with C8 and an output cap of1024 respecting EOS:

| Drafter / execution | K | Verification | Acceptance including target | TPOT ms |
| --- | ---: | --- | ---: | ---: |
| MLA original eager | 8 | standard | 3.8585 | 66.3418 |
| MLA optimized graphs | 8 | standard | 3.9091 | 7.5973 |
| MLA selected graphs | 8 | block | 3.8967 | 7.4988 |
| GQA contemporary control | 8 | block | 4.1662 | 6.4211 |

All rows are complete 300/300, zero errors, clean counter windows. K5 and K7
are separately recorded. The aggregate improvement is 8.85x over original
eager; MLA remains 16.8% slower than the contemporary GQA control. There is no
claim of exact GQA parity or that acceptance differences explain the entire
gap. Each configuration has one valid full run; variance was not estimated.

Graph/eager draft logits match exactly in 20 real batches across four D
workers. Fifteen changing-batch/padding graph cases, native reference/cache
checks, and a bounded existing block-sampler regression are also recorded.
The final original deployment completed 12 real requests across all six P
routes before/after idle. No private prompts, responses or weights are shipped.

## P and D launch examples

Install/import this branch with matching v0.31.0 native libraries before using
these commands. Model paths and existing RDMA devices are supplied by the
operator; these commands do not alter NICs, system services or other processes.
Run six producers with distinct GPU pairs, ports and bootstrap ports to form
6P1D. The router must register all six producers and the consumer.

```bash
export TARGET_MODEL=/models/glm53-mg46
export DRAFT_MODEL=/models/glm53-mla-step1200
export RDMA_DEVICES=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=BLHNC
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
```

One P instance, with PP2 fixed at **40/38**, PDCO, and K8:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_PP_LAYER_PARTITION=40,38 \
VLLM_DSPARK_PREFILL_CONTEXT_KV_ONLY=1 \
VLLM_MOONCAKE_BOOTSTRAP_PORT=8999 \
python -m vllm.entrypoints.cli.main serve "$TARGET_MODEL" \
  --host 0.0.0.0 --port 8111 --served-model-name glm53-mla-step1200 \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --max-model-len 1048576 --max-num-batched-tokens 16384 --max-num-seqs 8 \
  --gpu-memory-utilization 0.94 --block-size 64 --kv-cache-dtype fp8 \
  --no-enable-prefix-caching --enforce-eager \
  --compilation-config '{"cudagraph_mode":"NONE"}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":8,\"draft_tensor_parallel_size\":1,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false,\"kv_cache_dtype\":\"auto\",\"prefill_context_kv_only\":true}" \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"rdma\",\"device_name\":\"$RDMA_DEVICES\"}}" \
  --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice \
  --safetensors-load-strategy lazy --trust-remote-code
```

D on four distinct GPUs; full drafter, independent draft graphs:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
VLLM_DSPARK_PREFILL_CONTEXT_KV_ONLY=0 \
VLLM_SWA_PD_GROUP_SPLIT=1 \
VLLM_MOONCAKE_BOOTSTRAP_PORT=9007 \
python -m vllm.entrypoints.cli.main serve "$TARGET_MODEL" \
  --host 0.0.0.0 --port 8112 --served-model-name glm53-mla-step1200 \
  --tensor-parallel-size 1 --distributed-executor-backend mp \
  --data-parallel-size 4 --data-parallel-size-local 4 \
  --data-parallel-backend mp --enable-expert-parallel \
  --max-model-len 1048576 --max-num-batched-tokens 256 --max-num-seqs 32 \
  --gpu-memory-utilization 0.94 --block-size 64 --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[9,18,27,36,54,72,108,144,180,216,252]}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":8,\"draft_tensor_parallel_size\":1,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false,\"kv_cache_dtype\":\"auto\",\"prefill_context_kv_only\":false}" \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"rdma\",\"device_name\":\"$RDMA_DEVICES\"}}" \
  --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice \
  --safetensors-load-strategy lazy --trust-remote-code
```

The target MG46/1M serving configuration above is separate from the original
MG43/context196608 comparison. Its additional validation is recorded in the
PR evidence rather than relabeling old measurements as 1M results.

## Reproduce a fixed request evaluation

The portable benchmark preserves the executed request/counter logic, with
formatting and the repository-required `regex` import. Supply a JSONL file of
chat requests; private business inputs and responses are not included here.
Run on an otherwise idle service, so counter deltas do not include other users.

```bash
.venv/bin/python examples/deployment/glm53_mla_dspark/benchmark_pd.py \
  --base-url "$ROUTER_URL" --metrics-url "$DIRECT_DECODE_URL" \
  --model glm53-mla-step1200 --dataset "$REQUESTS_JSONL" \
  --output ./0831-300.json --count 300 --concurrency 8 \
  --max-tokens 1024 --seed 20260918 --num-spec-tokens 8
```

An optional `--manifest` can supply the P-role command snapshots for separate
producer-side completion/transfer checks. The original evaluations used these
snapshots. Preserve complete failed attempts as diagnostics and rerun the
whole request set; do not merge only successful rows from different attempts.

## Focused tests

The GPU test uses standard `unittest` and is also collectable by pytest:

```bash
.venv/bin/python -m unittest discover \
  -s tests/kernels/attention -p test_glm_dspark_mla.py -v
```

It checks dense FP32 references, window boundaries, strided multi-layer cache
views, released prefix pages, actual-token padding, empty/changing graph
batches, explicit noncausal-SWA rejection, and the shared kernel's ordinary
MHA full/windowed path. This area is already included by the attention-kernel
Buildkite group. End-to-end tests require the private checkpoint and are
reported separately; no fake public model registry example is added.
