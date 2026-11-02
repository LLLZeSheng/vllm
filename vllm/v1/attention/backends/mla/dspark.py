# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged DSpark MLA with an query-relative window and block masks."""

from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend


class DSparkMLAMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reorder_batch_threshold = 0

    def build(
        self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs
    ):
        return common_attn_metadata


class DSparkMLABackend(TritonMLABackend):
    @staticmethod
    def get_name():
        return "GLM_DSPARK_MLA"

    @staticmethod
    def get_builder_cls():
        return DSparkMLAMetadataBuilder
