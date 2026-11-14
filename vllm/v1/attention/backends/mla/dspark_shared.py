# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache registration and existing MLA operators for shared-history drafts."""

import math

import torch
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec


class SharedHistoryMLAMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reorder_batch_threshold = 0

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return common_attn_metadata


class SharedHistoryMLABackend(TritonMLABackend):
    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "DSPARK_SHARED_MLA"

    @staticmethod
    def get_builder_cls():
        return SharedHistoryMLAMetadataBuilder


class SharedHistoryMLACache(nn.Module, AttentionLayerBase):
    """One latent vector per token, aliased by sequential draft layers."""

    supports_dcp = False

    def __init__(
        self,
        config,
        num_heads: int,
        window: int,
        query_len: int,
        prefix: str,
        *,
        dtype: torch.dtype,
    ):
        super().__init__()
        current = get_current_vllm_config()
        self.layer_name = prefix
        self.num_heads = num_heads
        self.rank = config.kv_lora_rank
        self.rope_dim = config.qk_rope_head_dim
        self.head_size = self.rank + self.rope_dim
        self.window = window
        self.scale = (config.qk_nope_head_dim + self.rope_dim) ** -0.5
        sharing = config.shared_history_kv
        fp8 = sharing.get("fp8", True)
        requested = current.cache_config.cache_dtype
        if requested not in ("auto", "bfloat16", "float16", "fp8", "fp8_e4m3"):
            raise ValueError(f"Unsupported shared-history cache dtype: {requested}")
        if requested != "auto" and requested.startswith("fp8") != fp8:
            raise ValueError("KV cache dtype must match shared_history_kv.fp8")
        self.kv_cache_dtype = "fp8" if fp8 else "auto"
        self.dtype = torch.float8_e4m3fn if fp8 else dtype
        explicit_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        if (
            not fp8
            and requested in explicit_dtype
            and explicit_dtype[requested] != self.dtype
        ):
            raise ValueError("Non-FP8 KV cache must match the draft compute dtype")
        cache_scale = sharing.get("fp8_scale", 1.0)
        if not math.isfinite(cache_scale) or cache_scale <= 0:
            raise ValueError("shared_history_kv.fp8_scale must be finite and positive")
        self.register_buffer(
            "_k_scale", torch.tensor(cache_scale, dtype=torch.float32), persistent=False
        )
        self.kv_cache = torch.empty(0)
        if query_len <= 0:
            raise ValueError("Shared-history query length must be positive")
        self.query_len = query_len
        self.max_requests = current.scheduler_config.max_num_seqs
        self.native = (
            current.attention_config.backend == AttentionBackendEnum.FLASHMLA_SWA
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and fp8
            and self.rank == 512
            and self.rope_dim == 64
            and num_heads in (64, 128)
            and window >= 2048
            and self.query_len > 1
            and dtype == torch.bfloat16
            and math.frexp(cache_scale)[0] == 0.5
            and not envs.VLLM_BATCH_INVARIANT
        )
        if self.native:
            from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

            self.native = is_flashmla_sparse_supported()[0]
        registry = current.compilation_config.static_forward_context
        if prefix in registry:
            raise ValueError(f"Duplicate layer name: {prefix}")
        registry[prefix] = self

    def get_attn_backend(self):
        return SharedHistoryMLABackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        return SlidingWindowMLASpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.dtype,
            sliding_window=self.window,
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if kv_cache.ndim == 4 and kv_cache.shape[1] == 1:
            kv_cache = kv_cache.squeeze(1)
        if (
            kv_cache.ndim != 3
            or kv_cache.shape[-1] != self.head_size
            or kv_cache.dtype != self.dtype
        ):
            raise ValueError(
                "Shared-history cache geometry differs from the checkpoint"
            )
        self.kv_cache = kv_cache

    def store(self, latent, rotary, slots):
        if slots is None:
            return
        n = min(latent.shape[0], slots.numel())
        if n == 0:
            return
        cache = (
            self.kv_cache.view(torch.uint8)
            if self.dtype == torch.float8_e4m3fn
            else self.kv_cache
        )
        ops.concat_and_cache_mla(
            latent[:n].contiguous(),
            rotary[:n].contiguous(),
            cache,
            slots[:n],
            self.kv_cache_dtype,
            self._k_scale,
        )

    def reserve_workspace(self, device: torch.device):
        if self.native:
            from vllm.v1.attention.ops.windowed_flashmla import (
                windowed_flash_mla_workspace,
            )
            from vllm.v1.worker.workspace import is_workspace_manager_initialized

            if is_workspace_manager_initialized():
                windowed_flash_mla_workspace(
                    self.max_requests,
                    self.max_requests * self.query_len,
                    self.window,
                    self.query_len,
                    device,
                )

    def attend(self, query, metadata):
        if self.native:
            from vllm.v1.attention.ops.windowed_flashmla import windowed_flash_mla

            output, _ = windowed_flash_mla(
                query,
                self.kv_cache,
                metadata.block_table_tensor,
                metadata.seq_lens,
                metadata.query_start_loc,
                self.window,
                self.query_len,
                self._k_scale,
                self.scale,
                True,
            )
            return output
        requests = metadata.seq_lens.shape[0]
        if query.shape[0] != requests * self.query_len:
            raise ValueError("Shared-history DSpark requires uniform query blocks")
        splits = 4
        temporary = torch.empty(
            query.shape[0],
            self.num_heads,
            splits,
            self.rank + 1,
            dtype=torch.float32,
            device=query.device,
        )
        output = torch.empty(
            query.shape[0],
            self.num_heads,
            self.rank,
            dtype=query.dtype,
            device=query.device,
        )
        lse = torch.empty(query.shape[:2], dtype=torch.float32, device=query.device)
        keys = self.kv_cache.unsqueeze(-2)
        decode_attention_fwd(
            query,
            keys,
            keys[..., : self.rank],
            output,
            lse,
            metadata.block_table_tensor,
            metadata.seq_lens,
            temporary,
            splits,
            self.scale,
            page_size=self.kv_cache.shape[1],
            k_scale=self._k_scale,
            v_scale=self._k_scale,
            is_mla=True,
            sliding_window=self.window,
            query_len=self.query_len,
            causal=True,
        )
        return output
