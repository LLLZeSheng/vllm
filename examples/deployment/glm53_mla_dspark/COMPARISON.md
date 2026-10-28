# GLM-5.3 MLA DSpark: acceptance and TPOT comparison

Date: 2026-10-28.

## Measured result

**MG43 comparison selection: unchanged MLA step1200, K8, probabilistic drafts, block verification.**
Client-visible token-weighted TPOT improved from 66.3418 ms to 7.4988 ms (8.85x).
The contemporary GQA K8/block control is 6.4211 ms: MLA remains 16.8% slower,
with acceptance 3.8967 versus 4.1662, including one target token. This is accepted
as the deployment result, not described as exact GQA latency parity.

| Configuration | Spec tokens | Verification | Acceptance incl. target | TPOT (ms) | Successes |
| --- | ---: | --- | ---: | ---: | ---: |
| MLA, original eager | 8 | standard | 3.8585 | 66.3418 | 300/300 |
| MLA, initial full graphs | 8 | standard | 3.8331 | 7.7412 | 300/300 |
| MLA, optimized graphs/dispatch | 8 | standard | 3.9091 | 7.5973 | 300/300 |
| **MLA, selected deployment** | 8 | block | 3.8967 | 7.4988 | 300/300 |
| GQA, contemporary control | 8 | standard | 4.1316 | 6.5112 | 300/300 |
| GQA, contemporary control | 8 | block | 4.1662 | 6.4211 | 300/300 |
| MLA, K7 candidate | 7 | block | 3.8154 | 7.4848 | 300/300 |
| MLA, K5 candidate | 5 | block | 3.5231 | 7.6515 | 300/300 |

Each row is one complete clean 300-request run, zero errors. Run-to-run variance
was not estimated. The 0.2% K7/K8 TPOT difference is not established as a repeatable
gain; K5 was slower. The deployment was restored to K8. Weights were unchanged;
standard/block and different K values are explicitly separated.

With standard verification on both sides, TPOT changes from 66.3418 ms to7.5973ms,
an 8.73x improvement. The 8.85x aggregate result additionally selects block
verification; it is not attributed solely to a kernel change.

## Configuration and protocol

| Item | Value |
| --- | --- |
| Target | GLM-5.3 MG43, NVFP4 W4A4C8 |
| MLA draft | Supervised step1200; 5 causal SWA layers, window8192 |
| GQA draft | GQA 5SWA8192 step2100 control |
| Topology | Two hosts; six PP2 producers (40/38 layers); one TP1/DP4/EP4 consumer |
| P | PDCO; eager; max batched tokens16384; max sequences8; memory utilization0.94 |
| D | FULL_DECODE_ONLY graphs; max batched tokens256; max sequences32; memory utilization0.94 |
| Context limit | 196608 tokens; longest evaluated input148649; no1M validation claim |
| Cache | Target FP8; both contemporary draft controls BF16 |
| Draft sampling | Probabilistic; original temperature/top_p, tools and thinking fields retained |
| Prefix cache / adaptive verification | Both disabled |
| Evaluation | Same fixed0831-300, reservoir seed20260918; concurrency8 |
| Output | Original request cap, at most1024 tokens; natural EOS |
| Runtime | vLLM 0.31.0 native libraries + integration source; MRV2 |
| Packages | Python3.12; Torch2.13+cu130; FlashInfer0.7.0.post1 |
| Hardware | Device reports NVIDIA_H800, CUDA SM103a; about267.69GiB per GPU |

All configurations use identical 300 source indices and normalization rules.
Six missing/null tool.function.parameters fields become empty objects; no
samples or prompts are truncated/skipped. GQA and MLA are different trained
checkpoints, not a controlled attention-only architectural ablation. Their
different acceptance lengths contribute to latency but do not explain the
entire gap by themselves.

The historical 6.018 ms GQA result used 4SWA2048+1FA, FP8 draft KV, PP4 and a
different stack. It is not the matched control. The present GQA FP8/FA path did
not initialize successfully, so both contemporary drafts use BF16 caches.

## Metric definitions

Acceptance including target =1+accepted_draft_tokens/speculation_rounds from
D-only counter deltas. Position-i acceptance =accepted_at_i/speculation_rounds,
a prefix survival fraction, not conditional acceptance after reaching i.
Client token-weighted TPOT =sum(last_output_time-first_output_time)/
sum(completion_tokens-1), for requests with a first output and at least2 tokens.
This includes actual PD scheduling effects, not only CUDA kernel time.
Warmups, independent smokes and full evaluations have separate counter windows.

| Position | GQA / block | MLA / block | MLA / standard |
| ---: | ---: | ---: | ---: |
| 1 | 80.26% | 76.86% | 77.41% |
| 2 | 62.69% | 58.11% | 58.34% |
| 3 | 48.38% | 44.11% | 44.32% |
| 4 | 37.69% | 33.92% | 33.78% |
| 5 | 29.63% | 26.42% | 26.36% |
| 6 | 23.43% | 20.75% | 20.85% |
| 7 | 18.99% | 16.52% | 16.66% |
| 8 | 15.53% | 12.97% | 13.19% |

Sum the 8 position fractions and add 1 to reproduce each acceptance length.

## Implementation

- Register the GLM MLA drafter, configuration and weight mappings; preserve
  trained RMS/RoPE conventions and Markov proposal corrections. Latent 512 and
  rotary 64 are cached, total 576 BF16 elements per cached layer/token.
- Register per-layer SlidingWindowMLASpec and use native concat_and_cache_mla.
  P computes context KV only; existing PP auxiliary-state carry and Mooncake
  transfer feed the complete D-side drafter.
- Reuse the native Triton windowed MLA arithmetic from
  [PR #63795](https://github.com/vllm-project/vllm/pull/63795), pinned to
  `3de196f40700bf732ab260bf74883ad13c21f02a`. The GLM wrapper expands causal query metadata and masks padding;
  it does not introduce new attention arithmetic.
- Enable target and draft CUDA Graph execution. Previously the draft backend's
  NEVER capability disabled target graphs through the MRV2 global resolver.
  Handle shrinking/empty batches, NaN padding and released prefix pages safely.
- Give the draft its own request-multiple graph grid, separate from the target's
  K+1 verification-token grid. Use local dispatch only for the dense TP1/DCP1
  MLA drafter without EP collectives; retain target DP/EP synchronization.

No retraining, weight modification or new rejection-sampling algorithm was used.
The fork PR integrates these changes over the existing v0.31 deployment branch;
it is not another submission duplicating upstream[PR #62410](https://github.com/vllm-project/vllm/pull/62410).

## Validation and limits

- 15 graph-padding cases pass, including varying live batch sizes and empty ranks.
- 20 real long-context draft graph/eager checks across all four D workers: relative
  logitL2=0 and identical sampled draft tokens. K5 and K7 each have 20 checks too.
- Native arithmetic/dense-FP32 reference, window-boundary, real-weight and cache
  writer audits preserved. The bounded existing block sampler regression has 7
  passing cases; it is not represented as the entire upstream pytest suite.
- Every formal table row is300/300 successful with clean request/speculation
  counters. Final six-P and D health checks pass; no outstanding requests.
- 12 final real PD requests pass across all six P paths before/after7s idle.
  All4 D processes have graph capture receipts. Returned allocator records
  (12 P workers and D DP0) report zero OOM/retries; D HTTP memory RPC returns only
  DP0, so this is not an all-D-rank memory assertion.
- Both staged weight files match SHA256 `a4448468a48d2b534ecbee3f9dbdfde96ba8491ddcb9665c2d9dd4eac827957d`.

One initial MLA/block attempt hit a router-to-P connection reset. It is excluded
and preserved separately; the valid result reran all 300 requests, not selected
successful rows. HTTP keepalive increased from 5 to 600 seconds and router logging was reduced.
Idle close/reuse is a hypothesis, not a proven instrumented root cause.

The serving evidence belongs to vLLM 0.31.0 native libraries plus the integration
source, not an unmodified release or current upstream main. Validated target
PP2, draft TP1/PP1/DCP1, five causal 8192-token windows. No full1M-input, generic DCP or noncausal-SWA
claim. The later MG46/1M-config validation is separate below. No private weights, prompts, responses or infrastructure credentials are
part of this report or PR. AI assistance was used.

## Additional MG46 / 1M / exact-PR validation

The live target was subsequently changed to **MG46**, with PP2 fixed at
**40/38**, max model length **1,048,576**, P batch tokens16K, D256,
memory utilization0.94, the same MLA step1200 weights and K8/block/PDCO.

| Target / run | Max model length | Acceptance incl. target | TPOT ms | Requests |
| --- | ---: | ---: | ---: | ---: |
| MG43, original controlled comparison | 196608 | 3.8967 | 7.4988 | 300/300 |
| MG46, exact PR package | 1048576 | 3.8388 | 7.4770 | 300/300 |

All 3187 tracked vLLM package files staged on both hosts match the PR worktree;
matching v0.31.0 native libraries are retained. The packaged GPU unittest has 8
passing methods; 9 existing CPU regressions and 6 repository hook scripts pass.
Fourteen default computation paths are AST-equivalent to the measured source,
after removing disabled private diagnostics and relocating identical native
arithmetic into the shared kernel module. The MG46 run repeats 20 real
graph/eager comparisons across 4 D processes: logit L2=0, identical tokens.
The additional full 300 run has zero errors and clean counters.

| Representative worker | KV budget GiB | Physical KV pool GiB | Single 1M worst-case KV GiB | Capacity |
| --- | ---: | ---: | ---: | ---: |
| P / PP0 | 35.800 | 33.415 | 25.363 | 1.317 |
| P / PP1 | 31.316 | 31.315 | 23.769 | 1.317 |
| D / DP0 | 105.005 | 105.003 | 46.972 | 2.235 |

Capacity=physical blocks/cache-manager worst-case request blocks, including
SWA in-flight reservation. Every producer passes the 1M capacity check. These
are service-limit/cache-planning checks, **not a full 1M-input performance
benchmark**: the original 300 inputs remain at most 148649 tokens. Target and
maximum context both changed, so this row is not an isolated kernel ablation
or a matched comparison against the original MG43 GQA control.
