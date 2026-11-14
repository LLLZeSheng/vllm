# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import Qwen3Config

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.models.dspark_shared_mla import SharedHistoryEncoder, rotary

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "layers,hidden,rank,rope_dim", [(2, 48, 24, 8), (5, 64, 32, 16), (3, 96, 48, 32)]
)
@pytest.mark.parametrize("neox", [True, False])
def test_shared_encoder_zero_query_delta_preserves_the_historical_basis(
    layers, hidden, rank, rope_dim, neox
):
    """Every layer starts in the stored history's basis, at arbitrary dimensions."""
    config = Qwen3Config(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=rope_dim,
        kv_lora_rank=rank,
        qk_rope_head_dim=rope_dim,
        rms_norm_eps=1e-5,
        is_neox_style=neox,
        rope_parameters={"rope_type": "default", "rope_theta": 8000000.0},
        shared_history_kv={"encoder_type": "cross_layer_latent_mixture_v1"},
    )
    torch.manual_seed(42)
    encoder = SharedHistoryEncoder(config)
    delta = torch.nn.Linear(hidden, rank + rope_dim, bias=False)
    torch.nn.init.zeros_(delta.weight)
    states = torch.randn(3, hidden)
    positions = torch.tensor([0, 8193, 131071])
    history = encoder.history(states, positions)
    query = encoder.query(states, delta, encoder.kv_a_layernorm.weight, positions)
    for actual, expected in zip(query, history):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert history[0].shape == (3, rank)
    assert history[1].shape == (3, rope_dim)
    with torch.no_grad():
        delta.weight.fill_(0.1)
    for actual, expected in zip(encoder.history(states, positions), history):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rope_dim", [8, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shared_encoder_rope_layouts_agree_after_interleaving(rope_dim, dtype):
    """NeoX and interleaved exports differ only in how coordinate pairs are laid out."""
    torch.manual_seed(24)
    states = torch.randn(4, 2, rope_dim, dtype=dtype)
    positions = torch.tensor([0, 17, 8193, 131071])

    def interleave(x):
        a, b = x.chunk(2, -1)
        return torch.stack([a, b], -1).flatten(-2)

    expected = interleave(rotary(states, positions, 8000000.0, True))
    actual = rotary(interleave(states), positions, 8000000.0, False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], interleave(states)[0], rtol=0, atol=0)


def test_shared_history_cache_binding_allocates_one_history_for_all_layers():
    from vllm.config import CacheConfig
    from vllm.v1.attention.backends.mla.dspark_shared import SharedHistoryMLACache
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.attn_utils import bind_kv_caches, get_shared_kv_cache_layers

    config = Qwen3Config(
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        shared_history_kv={"fp8": True, "fp8_scale": 1.0},
    )
    current = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(cache_dtype="fp8", block_size=64),
    )
    with set_current_vllm_config(current):
        layers = [
            SharedHistoryMLACache(
                config,
                4,
                64,
                8,
                f"model.layers.{i}.self_attn.attn",
                dtype=torch.bfloat16,
            )
            for i in range(3)
        ]
    owner = layers[0].layer_name
    for layer in layers[1:]:
        layer.kv_sharing_target_layer_name = owner
    assert get_shared_kv_cache_layers(current) == {
        layer.layer_name: owner for layer in layers[1:]
    }
    history = torch.zeros(2, 64, 48, dtype=torch.float8_e4m3fn)
    caches = bind_kv_caches(
        {owner: history},
        current.compilation_config.static_forward_context,
        KVCacheConfig(num_blocks=2, kv_cache_tensors=[], kv_cache_groups=[]),
        current,
    )
    assert len(caches) == 3
    assert len({layer.kv_cache.data_ptr() for layer in layers}) == 1
    assert all(
        layer.get_kv_cache_spec(current).page_size_bytes == 64 * 48 for layer in layers
    )
