# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vllm.v1.worker.gpu.spec_decode.dspark.gqa_context_codec import (
    GQAContextCodec,
    prefill_spans,
)


@pytest.mark.parametrize(("rank", "width"), [(512, 520), (640, 650)])
def test_codec_record_width_and_aligned_round_trip(rank, width):
    codec = GQAContextCodec(torch.zeros(768), torch.eye(768)[:, :rank])
    hidden = torch.zeros((64, 768), dtype=torch.bfloat16)
    hidden[:32, :rank] = 1.0
    wire = codec.encode(hidden)
    assert wire.dtype == torch.uint8
    assert wire.shape == (64, width)
    assert wire.untyped_storage().nbytes() == 64 * width
    torch.testing.assert_close(codec.decode(wire, hidden.dtype), hidden, atol=0, rtol=0)
    with pytest.raises(ValueError, match="Invalid GQA context codec records"):
        codec.decode(wire[:, :-1], hidden.dtype)


def test_codec_only_changes_rows_owned_by_prefill():
    codec = GQAContextCodec(torch.zeros(256), torch.eye(256)[:, :128])
    hidden = torch.ones((11, 256), dtype=torch.bfloat16)
    spans = prefill_spans(
        np.array([0, 4, 9, 11]),
        np.array([3, 5, 0]),
        np.array([True, False, False]),
    )
    output = codec.round_trip_prefill(hidden, spans)
    torch.testing.assert_close(output[:3, 128:], torch.zeros_like(output[:3, 128:]))
    torch.testing.assert_close(output[3:], hidden[3:], atol=0, rtol=0)


def test_codec_rejects_calibration_for_a_different_hidden_size(tmp_path):
    path = str(tmp_path / "codec.safetensors")
    save_file(
        {"mean": torch.zeros(256), "basis": torch.eye(256)[:, :128].contiguous()},
        path,
    )
    GQAContextCodec.from_file(path, hidden_size=256, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="Invalid GQA context codec calibration"):
        GQAContextCodec.from_file(path, hidden_size=128, device=torch.device("cpu"))


def test_gqa_codec_wire_is_full_history_and_native_groups_are_local():
    from vllm.config import CacheConfig, VllmConfig
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
    from vllm.v1.kv_cache_interface import (
        DSparkGQAContextCacheSpec,
        HiddenStateCacheSpec,
        MLAAttentionSpec,
        SlidingWindowSpec,
    )

    config = VllmConfig(cache_config=CacheConfig(block_size=64))
    config.cache_config.kv_cache_layout = "BLHNC"
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    native = SlidingWindowSpec(
        block_size=64,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.uint8,
        sliding_window=8192,
    )
    groups = get_kv_cache_groups(
        config,
        {
            "target": MLAAttentionSpec(
                block_size=64, num_kv_heads=1, head_size=576, dtype=torch.uint8
            ),
            "draft0": native,
            "draft1": native,
            "wire": DSparkGQAContextCacheSpec(
                block_size=64,
                num_kv_heads=1,
                head_size=520,
                dtype=torch.uint8,
                restored_layers=("draft0", "draft1"),
            ),
        },
    )
    by_layer = {name: group for group in groups for name in group.layer_names}
    assert by_layer["target"].enable_kv_transfer
    for name in ("draft0", "draft1"):
        assert not by_layer[name].enable_kv_transfer
        assert by_layer[name].kv_cache_spec.sliding_window == 8192
    wire = by_layer["wire"]
    assert wire.enable_kv_transfer
    assert isinstance(wire.kv_cache_spec, HiddenStateCacheSpec)
    assert wire.kv_cache_spec.page_size_bytes == 64 * 520
