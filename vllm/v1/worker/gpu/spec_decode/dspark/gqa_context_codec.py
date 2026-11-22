# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-width FP8 encoding of the hidden input to all draft KV projections."""

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

_GROUP_SIZE = 128
_FP8 = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8).max
_SCALE_DTYPE = torch.float16
_MIN_SCALE = torch.finfo(_SCALE_DTYPE).smallest_normal


class GQAContextCodec(nn.Module):
    """Encode one context vector and restore the input to the native K/V path."""

    def __init__(
        self, mean: torch.Tensor, basis: torch.Tensor, normalize_input: bool = False
    ):
        super().__init__()
        if (
            mean.ndim != 1
            or basis.ndim != 2
            or basis.shape[0] != mean.numel()
            or not 0 < basis.shape[1] <= mean.numel()
            or basis.shape[1] % _GROUP_SIZE
        ):
            raise ValueError("Invalid GQA context codec mean or basis shape")
        self.register_buffer("mean", mean.to(torch.float32), persistent=False)
        self.register_buffer(
            "basis", basis.to(torch.bfloat16).contiguous(), persistent=False
        )
        self.normalize_input = normalize_input
        self.rank = basis.shape[1]
        self.groups = self.rank // _GROUP_SIZE
        self.record_bytes = self.rank + self.groups * 2

    @classmethod
    def from_file(
        cls, path: str, *, hidden_size: int, device: torch.device
    ) -> "GQAContextCodec":
        tensors = load_file(path, device="cpu")
        mean, basis = tensors.get("mean"), tensors.get("basis")
        if (
            mean is None
            or basis is None
            or mean.shape != (hidden_size,)
            or not mean.is_floating_point()
            or not basis.is_floating_point()
            or not torch.isfinite(mean).all()
            or not torch.isfinite(basis).all()
        ):
            raise ValueError("Invalid GQA context codec calibration")
        normalized = tensors.get("normalize_input")
        normalize_input = False
        if normalized is not None:
            if normalized.ndim != 0 or normalized.dtype != torch.bool:
                raise ValueError("Invalid GQA codec input normalization flag")
            normalize_input = bool(normalized)
        return cls(mean, basis, normalize_input).to(device)

    def round_trip_prefill(
        self, states: torch.Tensor, spans: list[tuple[int, int]]
    ) -> torch.Tensor:
        if not spans:
            return states
        if len(spans) == 1 and spans[0] == (0, len(states)):
            return self.decode(self.encode(states), dtype=states.dtype)
        prompt = torch.cat([states[start:end] for start, end in spans])
        restored = self.decode(self.encode(prompt), dtype=states.dtype)
        with_restored_prefill = states.clone()
        offset = 0
        for start, end in spans:
            size = end - start
            with_restored_prefill[start:end] = restored[offset : offset + size]
            offset += size
        return with_restored_prefill

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        values = context.float()
        if self.normalize_input:
            values = values * torch.rsqrt(
                values.square().mean(dim=-1, keepdim=True) + 1e-6
            )
        centered = (values - self.mean).to(self.basis.dtype)
        latent = (centered @ self.basis).float().view(-1, self.groups, _GROUP_SIZE)
        scales = (
            (latent.abs().amax(-1) / _FP8_MAX)
            .clamp(min=_MIN_SCALE, max=torch.finfo(_SCALE_DTYPE).max)
            .to(_SCALE_DTYPE)
        )
        payload = (latent / scales.float().unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX)
        payload = payload.to(_FP8).view(torch.uint8).reshape(-1, self.rank)
        return torch.cat((payload, scales.view(torch.uint8)), dim=-1)

    def decode(self, wire: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if (
            wire.dtype != torch.uint8
            or wire.ndim != 2
            or wire.shape[1] != self.record_bytes
        ):
            raise ValueError("Invalid GQA context codec records")
        payload = wire[:, : self.rank].contiguous().view(_FP8).float()
        scales = wire[:, self.rank :].contiguous().view(_SCALE_DTYPE).float()
        latent = payload.view(-1, self.groups, _GROUP_SIZE) * scales.unsqueeze(-1)
        restored = latent.view(-1, self.rank).to(self.basis.dtype) @ self.basis.T
        return (restored.float() + self.mean).to(dtype)


def prefill_spans(
    query_starts: np.ndarray,
    remaining: np.ndarray,
    is_prefilling: np.ndarray,
) -> list[tuple[int, int]]:
    count = np.minimum(np.diff(query_starts)[: len(remaining)], remaining)
    return [
        (int(query_starts[i]), int(query_starts[i] + count[i]))
        for i in np.flatnonzero(is_prefilling & (count > 0))
    ]
