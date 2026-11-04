# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped metadata reuse and BF16-exact rotary kernels for GLM DSpark MLA."""

from contextvars import ContextVar
from functools import lru_cache, wraps

import torch

import vllm.envs as envs
from vllm.triton_utils import tl, triton

REUSE: ContextVar[dict | None] = ContextVar("glm_dspark_mla_reuse", default=None)


@lru_cache(maxsize=16)
def _parse_flags(value: str) -> frozenset[str]:
    result = frozenset(filter(None, value.split(",")))
    if result - {"indices", "planner", "trig", "rope", "store"}:
        raise ValueError(f"Unsupported GLM MLA optimization: {value!r}")
    return result


def enabled(name: str) -> bool:
    return envs.VLLM_GLM_DSPARK_MLA_FLASHMLA and name in _parse_flags(
        envs.VLLM_GLM_DSPARK_MLA_OPTIMIZATIONS
    )


def forward_scope(function):
    """Discard all reused tensors when this model invocation finishes."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        if not envs.VLLM_GLM_DSPARK_MLA_OPTIMIZATIONS:
            return function(*args, **kwargs)
        token = REUSE.set({})
        try:
            return function(*args, **kwargs)
        finally:
            REUSE.reset(token)

    return wrapped


@triton.jit
def _rotate_exact(x, r, c, s, negative, DTYPE: tl.constexpr):
    bits = r.to(tl.uint32, bitcast=True)
    r = (bits ^ tl.where(negative, 0x80000000, 0).to(tl.uint32)).to(
        tl.float32, bitcast=True
    )
    # Explicit instructions also preserve signed zero under compiler algebraic
    # rewrites. Each product must round to BF16 before the addition.
    left = (
        tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, c],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        .to(DTYPE)
        .to(tl.float32)
    )
    right = (
        tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[r, s],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        .to(DTYPE)
        .to(tl.float32)
    )
    result = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return result.to(DTYPE)


@triton.jit
def _rope_exact(
    X,
    COS,
    SIN,
    Y,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    HEADS: tl.constexpr,
    NEOX: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 64)
    paired = (d + 32) % 64 if NEOX else d ^ 1
    negative = d < 32 if NEOX else d % 2 == 0
    x = tl.load(X + row * XS0 + head * XS1 + d).to(tl.float32)
    r = tl.load(X + row * XS0 + head * XS1 + paired).to(tl.float32)
    trig_d = d % 32 if NEOX else d // 2
    c = tl.load(COS + row * 32 + trig_d).to(tl.float32)
    s = tl.load(SIN + row * 32 + trig_d).to(tl.float32)
    result = _rotate_exact(x, r, c, s, negative, X.dtype.element_ty)
    tl.store(Y + (row * HEADS + head) * 64 + d, result)


def trig(layer, x, positions):
    state = REUSE.get()
    key = ("trig", id(positions), layer.rope_theta, x.dtype, layer.is_neox_style)
    if enabled("trig") and state is not None and key in state:
        return state[key]
    inv = layer.rope_theta ** (
        -torch.arange(0, 64, 2, device=x.device, dtype=torch.float32) / 64
    )
    phase = positions.float().unsqueeze(-1) * inv
    result = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
    if enabled("trig") and state is not None:
        state[key] = result
    return result


def rope(layer, x, positions):
    cosine, sine = trig(layer, x, positions)
    if enabled("rope"):
        output = torch.empty_like(x, memory_format=torch.contiguous_format)
        _rope_exact[(x.shape[0], x.shape[1])](
            x,
            cosine,
            sine,
            output,
            x.stride(0),
            x.stride(1),
            x.shape[1],
            layer.is_neox_style,
            enable_fp_fusion=False,
        )
        return output
    if layer.is_neox_style:
        cosine = torch.cat([cosine, cosine], -1).unsqueeze(1)
        sine = torch.cat([sine, sine], -1).unsqueeze(1)
        a, b = x.chunk(2, -1)
        rotated = torch.cat([-b, a], -1)
    else:
        cosine = cosine.repeat_interleave(2, -1).unsqueeze(1)
        sine = sine.repeat_interleave(2, -1).unsqueeze(1)
        rotated = torch.stack([-x[..., 1::2], x[..., ::2]], -1).flatten(-2)
    return x * cosine + rotated * sine


@triton.jit
def _rope_store_exact(
    LATENT,
    R,
    COS,
    SIN,
    S,
    C,
    SCALE,
    LS0: tl.constexpr,
    RS0: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    BLOCK: tl.constexpr,
    NEOX: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(S + row)
    if slot >= 0:
        scale = tl.load(SCALE)
        base = C + (slot // BLOCK) * CS0 + (slot % BLOCK) * CS1
        x = tl.arange(0, 512)
        latent = tl.load(LATENT + row * LS0 + x).to(tl.float32) / scale
        latent = tl.minimum(tl.maximum(latent, -448.0), 448.0).to(tl.float8e4nv)
        tl.store(base + x, latent.to(tl.uint8, bitcast=True))
        scales = scale.to(tl.uint32, bitcast=True)
        b = tl.arange(0, 16)
        tl.store(base + 512 + b, ((scales >> ((b % 4) * 8)) & 255).to(tl.uint8))
        d = tl.arange(0, 64)
        paired = (d + 32) % 64 if NEOX else d ^ 1
        negative = d < 32 if NEOX else d % 2 == 0
        r = tl.load(R + row * RS0 + d).to(tl.float32)
        p = tl.load(R + row * RS0 + paired).to(tl.float32)
        trig_d = d % 32 if NEOX else d // 2
        cosine = tl.load(COS + row * 32 + trig_d).to(tl.float32)
        sine = tl.load(SIN + row * 32 + trig_d).to(tl.float32)
        rotary = (
            _rotate_exact(r, p, cosine, sine, negative, R.dtype.element_ty).to(
                tl.float32
            )
            / scale
        )
        rotary = tl.minimum(tl.maximum(rotary, -448.0), 448.0).to(tl.float8e4nv)
        rotary = (rotary.to(tl.float32) * scale).to(tl.bfloat16)
        bits = rotary.to(tl.uint16, bitcast=True)
        tl.store(base + 528 + d * 2, (bits & 255).to(tl.uint8))
        tl.store(base + 529 + d * 2, (bits >> 8).to(tl.uint8))


def store_projected(layer, z, positions, slots):
    latent = layer.kv_a_layernorm(z[:, :512])
    rotary = z[:, 512:]
    cosine, sine = trig(layer, rotary, positions)
    if slots is not None:
        cache = layer.attn.kv_cache
        n = min(len(z), slots.numel())
        if n:
            _rope_store_exact[(n,)](
                latent,
                rotary,
                cosine,
                sine,
                slots,
                cache,
                layer.attn._k_scale,
                latent.stride(0),
                rotary.stride(0),
                cache.stride(0),
                cache.stride(1),
                cache.shape[1],
                layer.is_neox_style,
                enable_fp_fusion=False,
            )
