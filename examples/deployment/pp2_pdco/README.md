# PP2 DSpark Prefill Context-KV Only (PDCO)

This deployment branch packages the previously executed vLLM 0.31.0 PP/PD
integration. The GQA branch contains runnable source changes, a configurable
launcher, CPU regressions, and historical aggregate measurements.

| Item | Configuration |
| --- | --- |
| Repository | `LLLZeSheng/vllm` |
| GQA integration branch | `deploy/pp2-dspark-pdco-v031` |
| Optional MLA integration branch | `deploy/pp2-dspark-pdco-mla-v031` |
| Base | `Ar-zhou/vllm-zh`, `deploy/pp4-dspark-swa-1p1d-v031`, commit `65bad2204be2c848652e6f2c1c888239212ed2f2` |
| P | TP1 / PP2; target layers 0–37 and 38–77, partition **38/40** |
| D | TP1 / DP4 / EP; four complete decode workers |
| Target KV | FP8, BLHNC, 64 tokens per block |
| GQA draft | BF16 weights, 64 Q heads, 16 KV heads, head dimension 128; FP8 KV |
| Draft placement | Context-only draft on final P stage; complete draft on every D worker |
| Sampling | Probabilistic drafting; GQA block verification; adaptive verification off |
| Prefix caching | Off |

This is a fork integration, not a claim that the unmodified upstream 0.31.0
wheel contains PP/PD support. The upstream GQA-only feature submission remains
[vllm-project/vllm#62923](https://github.com/vllm-project/vllm/pull/62923).
This packaging does not update that PR or open another upstream PR.

## Design

```text
Prefill GPUs                                      Decode GPUs
+------------------+    +---------------------+   +-----------------------+
| PP0: layers 0-37 | -> | PP1: layers 38-77   |   | DP0 ... DP3, TP1/EP  |
| target forward   |    | target forward      |   | complete target       |
| auxiliary HS ----+--->| FC + hidden norm    |   | complete DSpark       |
+------------------+    | K/V projections     |   | draft + verification  |
                        | K norm + RoPE       |   +-----------------------+
                        | context KV writes   |            ^
                        +---------------------+            |
                          target + draft context KV -------+
                                      Mooncake
```

P skips draft embeddings, query/output projections, attention output, MLP,
Markov/confidence heads, and LM head. It retains the context fusion and KV
generation path, including the per-head K normalization and rotary transform.
P performs no draft-token generation. D still generates and verifies drafts.
All draft layers retain their original globally named cache entries; PDCO
does not remove SWA layers or change their attention visibility.

The six source files listed in `evidence/source-manifest.json` preserve the
executed GQA integration with license/import/format normalization. Besides selective loading, they
include the query-buffer/slot-mapping capacity fix and the generalized SWA
group matching needed by both 4SWA+1FA and 5SWA layouts. The PP relay and
Mooncake per-layer registration/transfer logic come from the pinned base.

## Setup

Use a separate checkout/environment with the matching 0.31.0 native libraries
and its existing Mooncake dependencies. Do not replace the source directory
of a running service. The launcher puts this checkout first on `PYTHONPATH`;
there is no model-directory `runtime/` or `sitecustomize.py` requirement.

```bash
git clone --branch deploy/pp2-dspark-pdco-v031 \
  https://github.com/LLLZeSheng/vllm.git vllm-pp2-pdco
cd vllm-pp2-pdco
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements/lint.txt
# Install the matching v0.31.0 wheel/runtime dependencies, including Mooncake.
# Then use that same wheel for this fork's editable native-library setup:
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=/path/to/matching-vllm-0.31.0.whl \
  uv pip install --python .venv/bin/python -e . --no-deps --no-build-isolation
```

The source manifest pins the integration base. This branch intentionally
keeps its existing fork configuration schema; its GQA PDCO switch is the
P-only `VLLM_DSPARK_PREFILL_CONTEXT_KV_ONLY=1` environment variable.
The upstream feature branch uses `prefill_context_kv_only` in speculative
configuration instead. These are distinct entry points to the same design.

## Launch

Set paths to the actual target/draft checkpoints and an existing RDMA device.
This example starts one P and one D on six cards; a second P can use cards
2–3 with a different API port and bootstrap port to form 2P1D.

```bash
export TARGET_MODEL=/path/to/GLM-5.3-NVFP4-W4A4C8-MG46-BNT3/v1
export DRAFT_MODEL=/path/to/sft-wtv-step800
export RDMA_DEVICES=mlx5_bond_1,mlx5_bond_2

.venv/bin/python examples/deployment/pp2_pdco/launch.py p \
  --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" \
  --gpus 0,1 --rdma-devices "$RDMA_DEVICES" \
  --port 8111 --bootstrap-port 8999 --manifest artifacts/p.json

.venv/bin/python examples/deployment/pp2_pdco/launch.py d \
  --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" \
  --gpus 4,5,6,7 --rdma-devices "$RDMA_DEVICES" \
  --port 8112 --bootstrap-port 9007 --manifest artifacts/d.json
```

Run the commands in separate terminals/services. Add `--dry-run` to inspect
the full generated command and environment without importing vLLM, touching
GPUs, or creating a manifest. Add `--full-prefill-draft` to P for the baseline.
Use the base integration's Mooncake-aware router to send requests through P/D;
the D API alone does not arrange external prefill.

`--memory-audit` enables the read-only worker extension in `memory_audit.py`.
It exposes `memory_audit` through the worker's collective RPC and reports
deduplicated target/draft storage, block counts, physical pools, and layer
cache views. Invoke it through the existing deployment's worker RPC tooling;
this launcher does not expose an unauthenticated administration endpoint.

| Launcher default | P | D |
| --- | ---: | ---: |
| Maximum total context | 524,288 | 524,288 |
| Batch token budget | 16,384 | 128 |
| Maximum sequences | 8 | 32 |
| GPU memory utilization | 0.92 | 0.94 |
| Speculative tokens | 8 | 8 |
| PDCO | Enabled | Disabled |

Override these limits explicitly for other capacity experiments. A configured
1M context is not proof of admitting a 1M request. Multiple P instances and
remote P/D hosts need distinct ports/engine identities and matching draft,
block size, cache dtype, sampling, and layer groups. The launcher never
changes NIC configuration or stops existing GPU processes.

## Historical validation

All acceptance lengths include the target's correction/bonus token:
`1 + accepted_draft_tokens / draft_rounds`. TPOT is output-token-weighted
client latency. The measurements below are **previous results**, not new GPU
evaluations performed while packaging this branch.

| Route | Complete P draft | PDCO P draft | Acceptance length, complete → PDCO | TPOT, complete → PDCO |
| --- | ---: | ---: | --- | --- |
| GQA PP4, 0831-300 | 5.553 GiB | 0.585950 GiB | 4.206129 → 4.183596 | 6.1364 → 5.9190 ms |
| MLA PP2, 0831-300 | 5.918165 GiB | 0.384538 GiB | 3.741657 / 3.743017 → 3.782916 | 59.2517 / 65.4891 → 60.7389 ms |

The GQA PP2 38/40 memory matrix passed four short-request smoke tests per
scenario, including external KV consumption. It is **not** a new 300-request
acceptance result; the GQA acceptance comparison above was PP4.

GQA PP2 with the fully quantized NVIDIA target, 4SWA+1FA, historical P512,
utilization 0.98, GiB (not the launcher's MG46 P16384/0.92 default):

| P stage | Target weights | Draft weights, full / PDCO | KV pool, full / PDCO | Allocated blocks, full / PDCO |
| --- | ---: | ---: | ---: | ---: |
| PP0, 38 layers | 194.03 | 0 / 0 | 23.65 / 27.99 | 16,996 / 20,111 |
| PP1, 40 layers | 219.67 | 5.55 / 0.59 | 28.83 / 34.11 | 16,996 / 20,111 |

See [the complete historical memory tables](evidence/gqa-pp2-memory.md) for
the profile, per-block bytes, unused budgets, D figures, and capacity formulas.
The larger pool follows from freed weight budget; PDCO itself does not reduce
the number of bytes required for an individual context token.

The optional MLA branch preserves the already executed GLM MLA model/backend,
physical cache-stride/64-bit-offset fix, producer-only model, and TP auxiliary
hidden-state reduction fix. Its custom draft latent cache is BF16, not FP8,
and its prior implementation allocates full-length cache even where the
attention mask is sliding. Its launcher profile therefore uses eager execution
and standard verification. Switch **both** P and D to that branch and pass
`--draft-kind mla`; use a `GlmDSparkMLAForCausalLM` checkpoint. The auxiliary
capture fix matters to TP reference measurements, even though PP2 uses TP1.

## Checks

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python \
  examples/deployment/pp2_pdco/test_regressions.py
.venv/bin/python -m compileall -q examples/deployment/pp2_pdco
git diff --check
```

The CPU suite executes the selected-weight loader and cache-group function
from this checkout without importing a GPU engine. It checks fused K/V
placement, malformed checkpoint rejection, producer-only roles, PP2/PP4
group compatibility for four/five SWA layers, and launcher P/D separation.
It does not replace a new end-to-end GPU evaluation.

Source integration and documentation were organized with AI assistance.
