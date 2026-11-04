# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native 576/512 FlashMLA with exact causal-window indices and FP8 values.

V3's packed record is 512 E4M3 bytes, four FP32 scales, and 64 BF16 RoPE
values. RoPE is rounded through E4M3 before packing to preserve the original
fully-E4M3 model's numerical contract; there is no model/head zero padding.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata
from vllm.v1.attention.ops.glm_dspark_mla import REUSE, enabled


@triton.jit
def _store(
    LATENT,
    R,
    S,
    C,
    SCALE,
    LS0: tl.constexpr,
    RS0: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    BLOCK: tl.constexpr,
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
        # Store scales bytewise; every layer can have a different scalar.
        scales = scale.to(tl.uint32, bitcast=True)
        b = tl.arange(0, 16)
        tl.store(base + 512 + b, ((scales >> ((b % 4) * 8)) & 255).to(tl.uint8))
        r = tl.arange(0, 64)
        rope = tl.load(R + row * RS0 + r).to(tl.float32) / scale
        rope = tl.minimum(tl.maximum(rope, -448.0), 448.0).to(tl.float8e4nv)
        rope = (rope.to(tl.float32) * scale).to(tl.bfloat16).to(tl.uint16, bitcast=True)
        tl.store(base + 528 + r * 2, (rope & 255).to(tl.uint8))
        tl.store(base + 529 + r * 2, (rope >> 8).to(tl.uint8))


def store(latent, rotary, cache, slots, scale):
    n = min(latent.shape[0], slots.numel())
    if n:
        _store[(n,)](
            latent,
            rotary,
            slots,
            cache,
            scale,
            latent.stride(0),
            rotary.stride(0),
            cache.stride(0),
            cache.stride(1),
            cache.shape[1],
        )


@triton.jit
def _indices(
    STARTS,
    LENGTHS,
    TABLE,
    INDICES,
    ACTIVE,
    N: tl.constexpr,
    REQUESTS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    request = tl.full((), 0, tl.int32)
    for r in range(REQUESTS):
        if row >= tl.load(STARTS + r + 1):
            request += 1
    valid = request < REQUESTS
    request = tl.minimum(request, REQUESTS - 1)
    start = tl.load(STARTS + request)
    end = tl.load(STARTS + request + 1)
    length = tl.load(LENGTHS + request)
    valid = valid & (row >= start) & (row < end) & (length > 0)
    causal_end = length - (end - start) + row - start + 1
    count = tl.minimum(causal_end, WINDOW)
    count = tl.where(valid, tl.maximum(count, 0), 0)
    offset = tile * TILE + tl.arange(0, TILE)
    position = tl.maximum(causal_end - WINDOW, 0) + offset
    block = position // PAGE
    mask = valid & (offset < count) & (block < TABLE_WIDTH)
    page = tl.load(TABLE + request * TABLE_STRIDE + block, mask, other=0)
    physical = page * PAGE + position % PAGE
    tl.store(
        INDICES + row * WINDOW + offset, tl.where(mask, physical, -1), offset < WINDOW
    )
    if tile == 0:
        tl.store(ACTIVE + row, count)


def attention(
    q,
    cache,
    table,
    starts,
    lengths,
    causal,
    window,
    num_actual_tokens=None,
    kv_scale=None,
):
    assert causal and window == 8192
    assert q.shape[-1] == 576 and q.shape[1] == 64
    assert cache.shape[-1] == 656 and cache.dtype == torch.uint8
    assert cache.stride(0) % 656 == 0, (
        "FlashMLA packed block stride must align to its 656-byte record"
    )
    padded = q.shape[0]
    n = padded if num_actual_tokens is None else num_actual_tokens
    if n == 0:
        return torch.zeros(padded, 64, 512, device=q.device, dtype=q.dtype)
    # Treat each query as an independent row: its exact visible window is
    # supplied explicitly, including block-internal causality and graph padding.
    reuse = REUSE.get()
    key = (id(table), id(starts), id(lengths), n, window, cache.shape[1])
    index_key, planner_key = ("indices", key), ("planner", key)
    if enabled("indices") and reuse is not None and index_key in reuse:
        indices, active = reuse[index_key]
    else:
        indices = torch.empty(n, 1, window, device=q.device, dtype=torch.int32)
        active = torch.empty(n, device=q.device, dtype=torch.int32)
        _indices[(n, triton.cdiv(window, 256))](
            starts,
            lengths,
            table,
            indices,
            active,
            n,
            lengths.shape[0],
            table.stride(0),
            table.shape[1],
            cache.shape[1],
            window,
            256,
        )
        if enabled("indices") and reuse is not None:
            reuse[index_key] = indices, active
    # Equal layers may share a schedule, but the first layer of every forward
    # must run the live planner, including inside a captured CUDA graph.
    if enabled("planner") and reuse is not None and planner_key in reuse:
        meta = reuse[planner_key]
    else:
        meta, _ = get_mla_metadata()
        if enabled("planner") and reuse is not None:
            reuse[planner_key] = meta
    out, _ = flash_mla_with_kvcache(
        q=q[:n].contiguous().unsqueeze(1),
        k_cache=cache.unsqueeze(2),
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=meta,
        softmax_scale=0.0625,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=active,
    )
    out = out.squeeze(1)
    out = torch.where((active > 0)[:, None, None], out, 0)
    if padded != n:
        out = torch.nn.functional.pad(out, (0, 0, 0, 0, 0, padded - n))
    return out
