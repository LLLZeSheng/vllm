# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode, CompilationMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config

    # None re-runs backend auto-selection for the draft, which can pick a
    # different attention class than the target; fall back to the target's.
    draft_attention_backend = (
        speculative_config.attention_backend or vllm_config.attention_config.backend
    )

    draft_cache_config = (
        replace(
            vllm_config.cache_config,
            cache_dtype=speculative_config.kv_cache_dtype,
        )
        if speculative_config.kv_cache_dtype is not None
        else vllm_config.cache_config
    )
    # The dense DSpark attention backend does not support the target MLA FP8
    # cache layout. Its cache is small, so retain it in the native dtype.
    if draft_cache_config.cache_dtype != "auto":
        draft_cache_config = replace(draft_cache_config, cache_dtype="auto")

    use_pp = vllm_config.parallel_config.pipeline_parallel_size > 1
    draft_compilation_config = vllm_config.compilation_config
    if use_pp:
        # Keep the target CUDA graph enabled, but isolate the last-stage-local
        # draft from target graph padding and compile buckets.
        draft_compilation_config = replace(
            draft_compilation_config,
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
        )

    # The draft keeps a local TP1/PP1 execution topology, but its compile-cache
    # identity must retain the enclosing worker's DP index. Otherwise every DP
    # worker reads and writes rank_0_0 AOT artifacts in the same Inductor cache,
    # which can produce cross-device artifacts that fail on the next restart.
    draft_parallel_config = copy.copy(speculative_config.draft_parallel_config)
    draft_parallel_config.data_parallel_rank = (
        vllm_config.parallel_config.data_parallel_rank
    )
    draft_parallel_config.data_parallel_index = (
        vllm_config.parallel_config.data_parallel_index
    )

    draft_vllm_config = replace(
        vllm_config,
        parallel_config=draft_parallel_config,
        compilation_config=draft_compilation_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=draft_attention_backend,
        ),
        cache_config=draft_cache_config,
    )
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    if use_pp:
        for module in draft_model.modules():
            if hasattr(module, "do_not_compile"):
                module.do_not_compile = True

        # Eager draft attention still executes inside the target runner's
        # forward context. Publish its non-overlapping attention entries there.
        target_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )
        for layer_name, layer in (
            draft_vllm_config.compilation_config.static_forward_context.items()
        ):
            existing = target_forward_context.get(layer_name)
            if existing is not None and existing is not layer:
                raise ValueError(
                    f"Duplicate target/draft attention layer: {layer_name}"
                )
            target_forward_context[layer_name] = layer

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    # In PP mode the target's embedding is a PPMissingLayer on the last stage,
    # while this complete local draft owns real embedding/head weights.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None)
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            draft_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

        target_lm_head = get_target_lm_head(target_model, target_language_model)
        draft_lm_head = getattr(draft_model, "lm_head", None)
        if target_lm_head is not None and _should_share(
            draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
        ):
            if draft_lm_head is not None:
                del draft_model.lm_head
            draft_model.lm_head = target_lm_head

    return draft_model
