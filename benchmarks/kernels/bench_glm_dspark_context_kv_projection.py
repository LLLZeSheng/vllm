# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare separate and stacked GLM MLA context projections on a CUDA device."""

import argparse
import statistics

import torch
from safetensors import safe_open
from torch.nn import functional as F


def measure(fn, repeats: int) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / repeats)
    return statistics.median(times)


def rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    value = x.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
    return normalized.to(x.dtype) * weight.to(x.dtype)


def cache_value(z: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
    latent = rms(z[:, :512], norm_weight)
    rotary = z[:, 512:]
    positions = torch.arange(z.shape[0], device=z.device, dtype=torch.float32)
    frequencies = 8_000_000 ** (
        -torch.arange(0, 64, 2, device=z.device, dtype=torch.float32) / 64
    )
    phase = positions[:, None] * frequencies
    cosine = torch.cat([phase.cos(), phase.cos()], dim=-1).to(z.dtype)
    sine = torch.cat([phase.sin(), phase.sin()], dim=-1).to(z.dtype)
    rotated = torch.cat([-rotary[:, 32:], rotary[:, :32]], dim=-1)
    return torch.cat([latent, rotary * cosine + rotated * sine], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to the GLM MLA model.safetensors")
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 128, 4096])
    args = parser.parse_args()

    with safe_open(args.checkpoint, framework="pt", device="cpu") as reader:
        weights = [
            reader.get_tensor(f"layers.{i}.self_attn.kv_a_proj_with_mqa.weight")
            .to("cuda")
            .contiguous()
            for i in range(5)
        ]
        norm_weights = [
            reader.get_tensor(f"layers.{i}.self_attn.kv_a_layernorm.weight").to(
                "cuda"
            )
            for i in range(5)
        ]
        hidden_norm_weight = reader.get_tensor("hidden_norm.weight").to("cuda")
    stacked_weight = torch.cat(weights)
    assert stacked_weight.shape == (2880, 6144)
    print("tokens  mode       separate_us  stacked_us  speedup  cache_relative_l2")

    for tokens in args.tokens:
        x = rms(
            torch.randn(tokens, 6144, device="cuda", dtype=weights[0].dtype),
            hidden_norm_weight,
        )

        def separate():
            return [F.linear(x, weight) for weight in weights]

        def stacked():
            return F.linear(x, stacked_weight).view(tokens, 5, 576)

        expected = separate()
        actual = stacked()
        old_cache = [cache_value(expected[i], norm_weights[i]) for i in range(5)]
        new_cache = [cache_value(actual[:, i], norm_weights[i]) for i in range(5)]
        error = torch.cat(
            [(old_cache[i].float() - new_cache[i].float()).flatten() for i in range(5)]
        )
        reference = torch.cat([part.float().flatten() for part in old_cache])
        relative_l2 = (error.norm() / reference.norm().clamp_min(1e-8)).item()

        repeats = 500 if tokens <= 128 else 100
        eager_separate = measure(separate, repeats)
        eager_stacked = measure(stacked, repeats)

        separate_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(separate_graph):
            separate()
        stacked_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(stacked_graph):
            stacked()
        graph_separate = measure(separate_graph.replay, repeats)
        graph_stacked = measure(stacked_graph.replay, repeats)

        for mode, old, new in (
            ("eager", eager_separate, eager_stacked),
            ("graph", graph_separate, graph_stacked),
        ):
            print(
                f"{tokens:<7} {mode:<10} {old:>11.2f} {new:>11.2f} "
                f"{old / new:>8.2f} {relative_l2:>12.3g}"
            )


if __name__ == "__main__":
    main()
