# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark MLA with a shared historical encoder and layer-local query deltas."""

import torch
from torch import nn
from torch.nn import functional as F

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.models.qwen3 import Qwen3MLP
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)
from vllm.v1.attention.backends.mla.dspark_shared import SharedHistoryMLACache


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + eps
    )
    return normalized.to(x.dtype) * weight.to(x.dtype)


def rotary(x, positions, theta: float, neox: bool):
    dim = x.shape[-1]
    inverse = theta ** (
        -torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
    )
    phase = positions.float().unsqueeze(-1) * inverse
    if neox:
        cosine = torch.cat([phase.cos(), phase.cos()], -1).to(x.dtype).unsqueeze(1)
        sine = torch.cat([phase.sin(), phase.sin()], -1).to(x.dtype).unsqueeze(1)
        a, b = x.chunk(2, -1)
        rotated = torch.cat([-b, a], -1)
    else:
        cosine = phase.cos().repeat_interleave(2, -1).to(x.dtype).unsqueeze(1)
        sine = phase.sin().repeat_interleave(2, -1).to(x.dtype).unsqueeze(1)
        rotated = torch.stack([-x[..., 1::2], x[..., ::2]], -1).flatten(-2)
    return x * cosine + rotated * sine


class EncoderNorm(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))


class SharedHistoryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        sharing = config.shared_history_kv
        if sharing.get("encoder_type") != "cross_layer_latent_mixture_v1":
            raise ValueError("Unsupported shared-history encoder type")
        self.layers = config.num_hidden_layers
        self.rank = config.kv_lora_rank
        self.eps = config.rms_norm_eps
        self.theta = config.rope_parameters["rope_theta"]
        self.neox = getattr(config, "is_neox_style", True)
        rope_dim = config.qk_rope_head_dim
        if min(self.layers, self.rank, rope_dim) <= 0 or rope_dim % 2:
            raise ValueError(
                "Layer count/rank must be positive; RoPE dimension must be even"
            )
        if config.rope_parameters.get("rope_type", "default") != "default":
            raise ValueError(
                "Shared-history checkpoints currently require default RoPE"
            )
        self.encoder_proj = nn.Linear(
            config.hidden_size, self.layers * self.rank, bias=False
        )
        self.encoder_norm = EncoderNorm((self.layers, self.rank))
        self.mix = nn.Linear(self.layers * self.rank, self.rank, bias=False)
        self.rope_proj = nn.Linear(config.hidden_size, rope_dim, bias=False)
        self.kv_a_layernorm = EncoderNorm(self.rank)

    def project(self, x):
        components = self.encoder_proj(x).view(x.shape[0], self.layers, self.rank)
        components = rms(components, self.encoder_norm.weight, self.eps)
        return self.mix(components.flatten(1)), self.rope_proj(x)

    def history(self, x, positions):
        latent, rope = self.project(x)
        return rms(latent, self.kv_a_layernorm.weight, self.eps), rotary(
            rope[:, None], positions, self.theta, self.neox
        ).squeeze(1)

    def query(self, x, delta_projection, norm_weight, positions):
        latent, rope = self.project(x)
        delta = delta_projection(x)
        latent = rms(latent + delta[:, : self.rank], norm_weight, self.eps)
        rope = rotary(
            (rope + delta[:, self.rank :])[:, None], positions, self.theta, self.neox
        )
        return latent, rope.squeeze(1)


class SharedHistoryAttention(nn.Module):
    def __init__(self, config, layer_idx: int, prefix: str):
        super().__init__()
        window, self.causal = _resolve_layer_attention(config, layer_idx)
        if not self.causal or window is None or window <= 0:
            raise ValueError("Shared-history MLA requires causal sliding attention")
        self.sliding_window = window
        tp = get_tensor_model_parallel_world_size()
        if config.num_attention_heads % tp:
            raise ValueError("Attention heads must be divisible by draft TP size")
        self.num_heads = config.num_attention_heads // tp
        self.rank = config.kv_lora_rank
        self.nope = config.qk_nope_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.value_dim = config.v_head_dim
        self.theta = config.rope_parameters["rope_theta"]
        self.neox = getattr(config, "is_neox_style", True)
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = EncoderNorm(config.q_lora_rank)
        self.eps = config.rms_norm_eps
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank,
            config.num_attention_heads * (self.nope + self.rope_dim),
            bias=False,
            prefix=prefix + ".q_b_proj",
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, self.rank + self.rope_dim, bias=False
        )
        self.kv_a_layernorm = EncoderNorm(self.rank)
        self.kv_b_proj = ColumnParallelLinear(
            self.rank,
            config.num_attention_heads * (self.nope + self.value_dim),
            bias=False,
            prefix=prefix + ".kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.value_dim,
            config.hidden_size,
            bias=False,
            prefix=prefix + ".o_proj",
        )
        spec = get_current_vllm_config().speculative_config
        assert spec is not None and spec.num_speculative_tokens is not None
        query_len = spec.num_speculative_tokens + (
            0 if getattr(config, "sample_from_anchor", True) else 1
        )
        self.attn = SharedHistoryMLACache(
            config,
            self.num_heads,
            window,
            query_len,
            prefix + ".attn",
            dtype=self.q_a_proj.weight.dtype,
        )
        # Registered once by the model, not repeatedly under every decoder layer.
        object.__setattr__(self, "shared_history", None)

    def forward(self, positions, hidden_states):
        context = get_forward_context()
        if context.attn_metadata is None:
            self.attn.reserve_workspace(hidden_states.device)
            return torch.zeros_like(hidden_states)
        metadata, _, _, slots = get_attention_context(self.attn.layer_name)
        if metadata is None:
            return torch.zeros_like(hidden_states)
        latent, rope = self.shared_history.query(
            hidden_states,
            self.kv_a_proj_with_mqa,
            self.kv_a_layernorm.weight,
            positions,
        )
        self.attn.store(latent, rope, slots)
        q = F.linear(
            rms(self.q_a_proj(hidden_states), self.q_a_layernorm.weight, self.eps),
            self.q_b_proj.weight,
        ).view(-1, self.num_heads, self.nope + self.rope_dim)
        nope, rope = q.split([self.nope, self.rope_dim], -1)
        rope = rotary(rope, positions, self.theta, self.neox)
        up = self.kv_b_proj.weight.view(
            self.num_heads, self.nope + self.value_dim, self.rank
        )
        latent_q = torch.bmm(nope.transpose(0, 1), up[:, : self.nope]).transpose(0, 1)
        latent = self.attn.attend(
            torch.cat([latent_q, rope], -1).contiguous(), metadata
        )
        values = torch.bmm(latent.transpose(0, 1), up[:, self.nope :].transpose(1, 2))
        values = values.transpose(0, 1).reshape(-1, self.num_heads * self.value_dim)
        return self.o_proj(values)[0]


class SharedHistoryMLP(Qwen3MLP):
    def forward(self, x):
        gate_weight, up_weight = self.gate_up_proj.weight.chunk(2, dim=0)
        activation = self.act_fn(
            torch.cat([F.linear(x, gate_weight), F.linear(x, up_weight)], -1)
        )
        output = F.linear(activation, self.down_proj.weight)
        if self.down_proj.reduce_results and self.down_proj.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output


class SharedHistoryDecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config,
        *,
        config,
        layer_idx,
        cache_config=None,
        quant_config=None,
        prefix="",
    ):
        nn.Module.__init__(self)
        if quant_config is not None:
            raise ValueError(
                "Shared-history MLA currently requires unquantized draft weights"
            )
        self.hidden_size = config.hidden_size
        self.self_attn = SharedHistoryAttention(
            config, layer_idx, prefix + ".self_attn"
        )
        self.mlp = SharedHistoryMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=None,
            prefix=prefix + ".mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class SharedHistoryModel(Qwen3DSparkModel):
    decoder_layer_cls = SharedHistoryDecoderLayer

    def __init__(self, *, vllm_config, start_layer_id=0, prefix=""):
        parallel = vllm_config.parallel_config
        if (
            parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError("Shared-history MLA does not support context parallelism")
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        del self.mask_embedding
        config = self.config
        groups = config.shared_history_kv.get("groups", [0] * config.num_hidden_layers)
        if len(groups) != config.num_hidden_layers or len(set(groups)) != 1:
            raise ValueError("This architecture requires a single shared history group")
        windows = {layer.self_attn.sliding_window for layer in self.layers}
        if len(windows) != 1:
            raise ValueError("Shared layers must use the same sliding window")
        self.shared_history = SharedHistoryEncoder(config)
        if getattr(config, "enable_confidence_head", True):
            with_markov = getattr(config, "confidence_head_with_markov", True)
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + (config.markov_rank if with_markov else 0),
                prefix=prefix + ".confidence_head",
                bias=True,
                with_markov=with_markov,
            )
        owner = self.layers[0].self_attn.attn.layer_name
        for i, layer in enumerate(self.layers):
            object.__setattr__(layer.self_attn, "shared_history", self.shared_history)
            if i:
                layer.self_attn.attn.kv_sharing_target_layer_name = owner
        self.is_mla = True

    def _build_fused_kv_buffers(self):
        # The historical encoder is shared; the GQA per-layer buffers do not apply.
        pass

    def context_kv_modules(self):
        modules = [self.hidden_norm, self.shared_history, self.layers[0].self_attn.attn]
        if self.use_aux_hidden_state:
            modules.append(self.fc)
        return modules

    def precompute_and_store_context_kv(
        self, context_states, context_positions, context_slot_mapping=None
    ):
        context_states = rms(
            context_states, self.hidden_norm.weight, self.config.rms_norm_eps
        )
        latent, rope = self.shared_history.history(context_states, context_positions)
        slots = (
            context_slot_mapping[0]
            if isinstance(context_slot_mapping, (list, tuple))
            else context_slot_mapping
        )
        self.layers[0].self_attn.attn.store(latent, rope, slots)


class DSparkSharedHistoryMLAForCausalLM(Qwen3DSparkForCausalLM):
    model_cls = SharedHistoryModel
    has_own_embed_tokens = True
    has_own_lm_head = True

    def load_weights(self, weights):
        normalized = {}
        for name, weight in weights:
            name = name.removeprefix("model.")
            if name in normalized:
                raise ValueError(f"Duplicate shared-history checkpoint weight: {name}")
            normalized[name] = weight
        required = {"embed_tokens.weight", "lm_head.weight"}
        if missing := required - normalized.keys():
            raise ValueError(
                f"Shared-history draft must provide its own weights: {missing}"
            )
        return super().load_weights(normalized.items())
