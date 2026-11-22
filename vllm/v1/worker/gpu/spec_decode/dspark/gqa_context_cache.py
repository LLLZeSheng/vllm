# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-history records and windowed restoration for GQA DSpark P/D."""

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.extract_hidden_states import (
    CacheOnlyAttentionBackend,
    CacheOnlyAttentionMetadata,
    CacheOnlyAttentionMetadataBuilder,
    basic_cache,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import DSparkGQAContextCacheSpec, KVCacheLayout
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dspark.gqa_context_codec import GQAContextCodec


class _ContextMetadataBuilder(CacheOnlyAttentionMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return CacheOnlyAttentionMetadata(common_attn_metadata.slot_mapping)


class _ContextCacheBackend(CacheOnlyAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DSPARK_GQA_CONTEXT_CACHE"

    @classmethod
    def get_builder_cls(cls):
        return _ContextMetadataBuilder

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.BLHNC,)


class GQAContextCache(AttentionLayerBase):
    supports_dcp = False

    def __init__(
        self,
        config: VllmConfig,
        codec: GQAContextCodec,
        restored_layers: tuple[str, ...],
        window: int,
    ):
        layer_index = extract_layer_index(restored_layers[-1]) + 1
        self.layer_name = f"model.layers.{layer_index}.gqa_codec.cache"
        self.codec = codec
        self.restored_layers = restored_layers
        self.window = window
        self.restore_ends: dict[str, int] = {}
        registry = config.compilation_config.static_forward_context
        if self.layer_name in registry:
            raise ValueError("Duplicate GQA DSpark context cache")
        registry[self.layer_name] = self

    def get_attn_backend(self):
        return _ContextCacheBackend

    def get_kv_cache_spec(self, config: VllmConfig):
        return DSparkGQAContextCacheSpec(
            block_size=config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.codec.record_bytes,
            dtype=torch.uint8,
            max_tp_shards=1,
            dcp_sharded=False,
            restored_layers=self.restored_layers,
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if (
            kv_cache.dtype != torch.uint8
            or kv_cache.ndim != 4
            or kv_cache.shape[1] != 1
            or kv_cache.shape[-1] != self.codec.record_bytes
        ):
            raise ValueError("Allocated KV cache does not match the GQA codec format")
        super().bind_kv_cache(kv_cache)

    def store(
        self,
        states: torch.Tensor,
        slots: torch.Tensor,
        spans: list[tuple[int, int]],
    ) -> None:
        for start, end in spans:
            wire = self.codec.encode(states[start:end])
            basic_cache(wire.unsqueeze(1), self.kv_cache, slots[start:end])

    def restore(
        self,
        draft,
        input_batch: InputBatch,
        block_tables: BlockTables,
        wire_gid: int,
        native_gids: list[int],
    ) -> None:
        for row, req_id in enumerate(input_batch.req_ids):
            end = self.restore_ends.get(req_id, 0)
            if not end or not input_batch.is_prefilling_np[row]:
                continue
            computed = int(input_batch.num_computed_prefill_tokens_np[row])
            scheduled = int(input_batch.num_scheduled_tokens[row])
            # Wait for the final local prefill chunk. Its draft query is the
            # first used by the verifier, and earlier chunks can evict SWA pages.
            if computed + scheduled < input_batch.prefill_len_np[row]:
                continue
            self.restore_ends.pop(req_id)
            anchor = computed + scheduled
            begin = max(0, min(computed, anchor - self.window + 1))
            if begin >= end:
                continue
            get_kv_transfer_group().wait_for_layer_load(self.layer_name)
            positions = torch.arange(begin, end, device=self.kv_cache.device)
            wire_bs = self.kv_cache.shape[2]
            wire_blocks = block_tables.input_block_tables[wire_gid][
                row, positions // wire_bs
            ].long()
            if torch.any(wire_blocks == 0):
                raise RuntimeError("Missing GQA codec pages for remote draft context")
            wire = self.kv_cache[wire_blocks, 0, positions % wire_bs]
            restored = self.codec.decode(
                wire, dtype=draft.model.hidden_norm.weight.dtype
            )
            native_slots = []
            for gid in native_gids:
                bs = block_tables.kernel_block_sizes[gid]
                blocks = block_tables.input_block_tables[gid][
                    row, positions // bs
                ].long()
                native_slots.append(
                    torch.where(blocks > 0, blocks * bs + positions % bs, -1)
                )
            draft.precompute_and_store_context_kv(restored, positions, native_slots)
