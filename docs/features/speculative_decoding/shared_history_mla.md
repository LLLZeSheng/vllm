# Shared-history MLA DSpark drafts

A shared-history draft stores one historical MLA latent vector per token. Each
decoder layer interprets that vector through its own KV up-projection. The current
query block uses a layer-local delta in the same latent basis. Layers execute
sequentially, so their query slots can be reused without overwriting history.

This is a checkpoint architecture, not an inference-time conversion of arbitrary
GQA or MLA weights. It requires weights trained with the shared historical
encoder and the layer-local query deltas.

## Checkpoint configuration

Use a standalone Hugging Face configuration declaring
`DSparkSharedHistoryMLAForCausalLM`. The older
`GlmDSparkSharedHistoryMLAForCausalLM` spelling is an alias. Configuration and
checkpoint paths need no naming convention, and no checkpoint-side Python files
are required.

The ordinary decoder fields specify hidden size, intermediate size, layer count,
query heads, activation and normalization epsilon. MLA dimensions come from
`q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, and
`v_head_dim`. For example, the additional shared-history fields are:

```json
{
  "architectures": ["DSparkSharedHistoryMLAForCausalLM"],
  "model_type": "qwen3",
  "block_size": 8,
  "sample_from_anchor": true,
  "mask_token_id": 123,
  "use_aux_hidden_state": true,
  "eagle_aux_hidden_state_layer_ids": [2, 21, 39, 57, 76],
  "is_neox_style": true,
  "is_causal": true,
  "use_sliding_window": true,
  "sliding_window": 8192,
  "shared_history_kv": {
    "encoder_type": "cross_layer_latent_mixture_v1",
    "groups": [0, 0, 0, 0, 0],
    "fp8": true,
    "fp8_scale": 1.0
  }
}
```

This is a configuration fragment, not a complete model configuration. Set the
mask token and auxiliary layers to the values used in training. Auxiliary states
are concatenated and passed through the trained FC; widths are checked, not
padded or truncated. Legacy `aux_hidden_state_layer_ids` is accepted when the
Eagle-specific field is absent.

All draft layers must belong to one history group and use the same causal
sliding window. Layer count and MLA dimensions are not fixed to the example.
Default RoPE supports NeoX and interleaved coordinate layouts. Other scaling
schemes, noncausal layers, mixed windows, quantized draft weights and context
parallelism and batch-invariant execution are not supported by this architecture.

The checkpoint supplies its own embedding and vocabulary head. The remaining
weights use the DSpark names: `fc`, `hidden_norm`, `layers`, `norm`,
`markov_head`, and optional `confidence_head`. Shared weights are registered only
under `shared_history`: `encoder_proj`, `encoder_norm`, `mix`, `rope_proj`, and
`kv_a_layernorm`. Query deltas retain each layer's `kv_a_proj_with_mqa` and
`kv_a_layernorm` names. Both root-level and `model.`-prefixed keys are accepted.

## Serving

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /path/to/target \
  --speculative-config '{"method":"dspark","model":"/path/to/draft","num_speculative_tokens":8,"kv_cache_dtype":"fp8"}'
```

The model reuses DSpark's Markov sampling, verification and auxiliary-state
handling. It registers cache aliases through the existing KV-sharing interface.
Only the first draft layer owns the historical cache. The shared encoder writes
that history once; each layer writes its own query vectors into the same
transient slots before attending.

`prefill_context_kv_only` uses the normal DFlash/DSpark mechanism to retain only
the FC, hidden normalization and shared history encoder on a prefill worker.
There is no separate context-only architecture or P/D connector implementation.
The prefill and decode instances must use matching checkpoint/configuration,
cache precision and native-versus-Triton selection.

## Attention and storage

Persistent KV uses the standard compressed layout: `kv_lora_rank +
qk_rope_head_dim` elements per token, shared across the decoder layers. FP8 cache
storage uses E4M3 and the checkpoint's fixed positive scalar scale. Non-FP8
checkpoints use their compute dtype. Cache precision must match the checkpoint's
`shared_history_kv.fp8` setting; `auto` resolves from that setting.

Select `attention_backend: "FLASHMLA_SWA"` in the speculative configuration to
use the existing windowed FlashMLA operator when its kernel supports the
geometry, device, precision, window, scale and query width. Other cases use the
existing Triton decode operator. Dimensions and heads are never zero-padded to
make an unsupported geometry fit a native kernel. No new CUDA or Triton kernels
are introduced here.

The native path temporarily packs the visible window. Its workspace is reserved
during profiling and reused across layers; it does not create another persistent
history per layer. For latent512/RoPE64, the persistent FP8 record is 576 bytes;
temporary native records are 656 bytes. Include workspace and allocator alignment
when measuring actual memory. See [the FlashMLA backend documentation](../../design/attention_backends.md)
for the backend's capability limits.

Gate and up projections remain separate GEMMs to preserve this architecture's
training forward. Changing their GEMM shapes or switching between Triton and
native attention can change BF16 rounding. A cache-sharing check alone is not
proof of full-logit equivalence: compare the complete forward, including the
vocabulary and Markov heads, for the deployment being used.
