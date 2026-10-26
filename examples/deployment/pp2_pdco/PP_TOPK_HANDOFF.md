# GLM DSA Top-K Handoff Across Pipeline Stages

## Scope and provenance

This fork supports PP boundaries inside a sparse-attention index-sharing
group, including a **40/38** partition of a 78-layer GLM DSA target.
Layer numbers below are zero-based.

The production CUDA implementation is
`vllm/models/deepseek_v32/nvidia/model.py` (`DeepseekV32Model`).
`GlmMoeDsaForCausalLM` selects that implementation on CUDA. It is not the
similarly named `Glm4MoeLiteModel` implementation.

**The core relay already exists in the integration base.** Its original
implementation was introduced in
[Ar-zhou/vllm-zh, e9ed5b6](https://github.com/Ar-zhou/vllm-zh/commit/e9ed5b64addeba164b2e8c046e7d4907a56846d5)
and is retained by the pinned `65bad2204be2c848652e6f2c1c888239212ed2f2`
integration. This branch adds an explicit deployment option and regression
coverage; it does not introduce a competing index-relay mechanism or copy
the unrelated structured-output workaround from that older commit.

## Why 40/38 needs the relay

For `index_topk_freq=4` and `index_skip_topk_offset=3`, layers 38 and 42
compute new Top-K indices; layers 39, 40 and 41 reuse the indices from 38.

| Partition | PP0 layers | PP1 layers | First layer of PP1 |
| --- | --- | --- | --- |
| 38/40 | 0–37 | 38–77 | Computes new indices |
| 40/38 | 0–39 | 40–77 | Reuses indices computed on PP0 |

```text
PP0: layers 0 ... 38 [compute Top-K] ... 39
       |
       | hidden_states + residual + topk_indices (int32)
       v
PP1: copy received indices into its own persistent Top-K buffer
     layer 40 [reuse] -> 41 [reuse] -> 42 [compute new Top-K] ... 77
```

The receiving stage must preserve the layer-38 result. Recomputing an
indexer at layer 40 would change the model rather than repair its relay.

## Implementation contract

1. Keep the stage's shared `topk_indices_buffer` on the model instance.
2. Include the active rows as `topk_indices` in outgoing
   `IntermediateTensors`, along with the existing hidden states, residual
   and any auxiliary hidden-state entries.
3. Allocate receive buffers with shape `[batch_size, index_topk]` and
   **int32** dtype, independently of the hidden-state dtype.
4. Before the receiving stage's first decoder layer runs, copy the received
   rows into its persistent Top-K buffer. The copy is included in CUDA
   Graph capture and is replayed for every batch.
5. Preserve the custom receive-buffer factory when auxiliary feature
   extraction is enabled. The current integration uses `EagleModelMixin`
   without replacing this factory.

These are **logical token indices**, not physical KV block addresses.
Each stage maps them through its own block table. This PP activation relay
is separate from Mooncake's P-to-D transfer of target MLA/DSA KV caches.
It changes neither checkpoint weights nor the target's index-sharing
pattern, and is independent of whether the draft method is MTP or DSpark.

## Deployment

The existing DSpark PDCO example now accepts:

```bash
.venv/bin/python examples/deployment/pp2_pdco/launch.py p \
  --target /path/to/target --draft /path/to/draft \
  --gpus 0,1 --rdma-devices mlx5_0 \
  --pp-partition 40,38 --dry-run
```

The default remains `38,40`. Only P uses the partition; D still removes
`VLLM_PP_LAYER_PARTITION` from its environment. Before a real launch, the
example checks that the two positive layer counts cover the target's
`num_hidden_layers`.

For the existing MTP launcher, use the same CUDA model implementation and
set `VLLM_PP_LAYER_PARTITION=40,38` on P. This example launcher remains a
DSpark launcher; it does not implement the separate MTP KV-only package.

## Validation and limits

The standalone regression suite executes the **production forward and
receive-buffer factory methods**, loaded without heavyweight kernel
imports, against deterministic toy layers:

```bash
.venv/bin/python examples/deployment/pp2_pdco/test_pp_topk_handoff.py -v
.venv/bin/python examples/deployment/pp2_pdco/test_regressions.py -v
```

Coverage includes 38/40, 40/38 and a group spanning three stages; successive
batch lengths and positions; auxiliary hidden-state relay; int32 index
buffers with BF16 hidden states; and a negative control where dropping
the indices changes the output. When CUDA is available, the suite also
checks that graph replay uses fresh received indices after changing the
input and poisoning the receiving buffer.

The CUDA Graph regression runs on **one GPU with toy layers**. It is not
a full-model, multi-GPU PP evaluation. Existing serving processes and
their 38/40 partition were not changed for this submission. Real MG46
40/38 acceptance length, memory balance and TTFT still require a separate
deployment evaluation; no such performance result is claimed here.

Recorded results on 2026-10-26:

| Check | Environment | Result |
| --- | --- | --- |
| Relay suite on CPU | Python 3.12, PyTorch 2.13.0+cu130, CUDA disabled | 5 passed, CUDA case skipped |
| Relay suite including CUDA Graph | NVIDIA H800, PyTorch 2.13.0+cu130 | 6 passed |
| Existing PDCO regression suite | CPU | 5 passed |

The CUDA run's raw result is saved in
[`evidence/pp-topk-handoff-tests.json`](evidence/pp-topk-handoff-tests.json).
These standalone tests reuse the existing environment's PyTorch packages;
they do not install a different vLLM into the running serving environment.
