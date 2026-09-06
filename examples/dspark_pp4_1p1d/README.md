# GLM-5.2 DSpark PP4 + 1P1D command templates

Set paths and network addresses for the deployment first:

```bash
export TARGET=/path/to/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1
export DRAFT=/path/to/dspark/step400
export P_HOST=192.168.0.68
export D_HOST=192.168.0.55
```

Run the same vLLM commit on P and D. NIXL metadata from this branch is version 9
and is intentionally incompatible with an unpatched peer.

## P: TP1 / PP4

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_NIXL_SIDE_CHANNEL_HOST="${P_HOST}"
export VLLM_NIXL_SIDE_CHANNEL_PORT=5800

vllm serve "${TARGET}" \
  --host 0.0.0.0 --port 8100 \
  --served-model-name glm-5.2-dspark \
  --tensor-parallel-size 1 --pipeline-parallel-size 4 \
  --max-model-len 131072 --max-num-batched-tokens 2112 \
  --kv-cache-dtype bfloat16 --block-size 64 \
  --attention-backend FLASHINFER_MLA_SPARSE \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config \
  "{\"model\":\"${DRAFT}\",\"method\":\"dspark\",\"num_speculative_tokens\":8,\"draft_sample_method\":\"greedy\",\"attention_backend\":\"FLASH_ATTN\"}" \
  --kv-transfer-config \
  "{\"kv_connector\":\"NixlPushConnector\",\"kv_role\":\"kv_producer\",\"engine_id\":\"P-$(date +%s)\"}"
```

## D: TP1 / DP8

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_NIXL_SIDE_CHANNEL_HOST="${D_HOST}"
export VLLM_NIXL_SIDE_CHANNEL_PORT=5859

vllm serve "${TARGET}" \
  --host 0.0.0.0 --port 8200 \
  --served-model-name glm-5.2-dspark \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --data-parallel-size 8 --data-parallel-size-local 8 \
  --data-parallel-multi-port-external-lb \
  --data-parallel-supervisor-port 8299 \
  --max-model-len 131072 --max-num-batched-tokens 2112 --max-num-seqs 1 \
  --kv-cache-dtype bfloat16 --block-size 64 \
  --attention-backend FLASHINFER_MLA_SPARSE \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config \
  "{\"model\":\"${DRAFT}\",\"method\":\"dspark\",\"num_speculative_tokens\":8,\"draft_sample_method\":\"greedy\",\"attention_backend\":\"FLASH_ATTN\"}" \
  --kv-transfer-config \
  "{\"kv_connector\":\"NixlPushConnector\",\"kv_role\":\"kv_consumer\",\"engine_id\":\"D-$(date +%s)\",\"kv_connector_extra_config\":{\"push_registration_timeout\":300}}"
```

The routing layer must send a request through P first and then route the same
transfer identifier to D. The routing implementation is deployment-specific and
is not part of this patch.

## Metrics

Aggregate all DP engine endpoints, rather than reading only engine 0:

```text
vllm:spec_decode_num_drafts_total
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_accepted_tokens_total
vllm:request_time_per_output_token_seconds
```

Pure DSpark average accepted length is:

```text
delta(spec_decode_num_accepted_tokens_total)
------------------------------------------------
delta(spec_decode_num_drafts_total)
```
