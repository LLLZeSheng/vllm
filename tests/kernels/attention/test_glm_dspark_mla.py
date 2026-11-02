# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise causal windows and live-batch changes in the MLA graph path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.model_executor.models.glm_dspark_mla import DSparkMLACache
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.dspark_upstream_window import attention
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd


class TestGlmDSparkMLACacheSpec(unittest.TestCase):
    def test_cache_spec_honors_bf16_and_fp8_storage(self):
        config = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))
        for window in [None, 8192]:
            for dtype in ["auto", "bfloat16", "fp8", "fp8_e4m3"]:
                with self.subTest(window=window, dtype=dtype):
                    cache = SimpleNamespace(kv_cache_dtype=dtype, sliding_window=window)
                    spec = DSparkMLACache.get_kv_cache_spec(cache, config)
                    fp8 = dtype in ("fp8", "fp8_e4m3")
                    self.assertEqual(
                        spec.dtype, torch.float8_e4m3fn if fp8 else torch.bfloat16
                    )
                    self.assertEqual(spec.page_size_bytes, 64 * 576 * (1 if fp8 else 2))

    def test_unsupported_fp8_format_is_rejected(self):
        cache = SimpleNamespace(kv_cache_dtype="fp8_e5m2", sliding_window=8192)
        config = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))
        with self.assertRaisesRegex(ValueError, "E4M3"):
            DSparkMLACache.get_kv_cache_spec(cache, config)

    def test_context_only_producer_accepts_fp8_without_changing_config(self):
        from vllm.model_executor.models.dspark_context_kv import MLAContextKVAttention

        config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_attention_heads=64,
            rope_parameters={"rope_theta": 8000000},
            hidden_size=6144,
            rms_norm_eps=1e-5,
        )
        for dtype in ["bfloat16", "fp8", "fp8_e4m3"]:
            with self.subTest(dtype=dtype):
                cache_config = SimpleNamespace(cache_dtype=dtype)
                with (
                    patch(
                        "vllm.model_executor.models.dspark_context_kv."
                        "_resolve_layer_attention",
                        return_value=(8192, True),
                    ),
                    patch(
                        "vllm.model_executor.models.glm_dspark_mla.DSparkMLACache",
                        return_value=torch.nn.Identity(),
                    ),
                ):
                    MLAContextKVAttention(config, 0, cache_config, "model.layers.0")
                self.assertEqual(cache_config.cache_dtype, dtype)


@unittest.skipUnless(current_platform.is_cuda(), "CUDA is required")
class TestGlmDSparkMLA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def inputs(self, lengths, query_lengths):
        pages = sum((n + 63) // 64 for n in lengths) + 3
        backing = torch.randn(pages, 64, 5, 576, device="cuda", dtype=torch.bfloat16)
        cache = backing[:, :, 3, :]
        table = torch.full(
            (len(lengths), (max(lengths) + 63) // 64 + 1),
            -1,
            device="cuda",
            dtype=torch.int32,
        )
        permutation = torch.randperm(pages, device="cuda", dtype=torch.int32)
        offset = 0
        for i, n in enumerate(lengths):
            count = (n + 63) // 64
            table[i, :count] = permutation[offset : offset + count]
            offset += count
        query = (
            torch.randn(
                sum(query_lengths), 64, 576, device="cuda", dtype=torch.bfloat16
            )
            * 0.5
        )
        cumulative = [0]
        for n in query_lengths:
            cumulative.append(cumulative[-1] + n)
        starts = torch.tensor(cumulative, device="cuda", dtype=torch.int32)
        lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        return query, cache, table, starts, lens

    def dense(self, args, lengths, query_lengths, window, causal=True):
        q, cache, table, _, _ = args
        result = []
        offset = 0
        for i, (length, count) in enumerate(zip(lengths, query_lengths)):
            for j in range(count):
                end = length - count + j + 1 if causal else length
                begin = max(0, end - window) if window else 0
                positions = torch.arange(begin, end, device="cuda")
                keys = cache[table[i, positions // 64].long(), positions % 64].float()
                scores = q[offset + j].float() @ keys.T * (256**-0.5)
                result.append(scores.softmax(-1) @ keys[:, :512])
            offset += count
        return torch.stack(result).to(q.dtype)

    def assert_reference(self, actual, expected):
        self.assertTrue(bool(actual.isfinite().all()))
        error = (actual.float() - expected.float()).norm()
        relative = error / expected.float().norm().clamp_min(1e-8)
        self.assertLess(relative.item(), 0.008)

    def test_causal_window_boundaries(self):
        for length in [13, 8191, 8192, 8193, 16407]:
            with self.subTest(length=length):
                args = self.inputs([length], [8])
                expected = self.dense(args, [length], [8], 8192)
                self.assert_reference(attention(*args, True, 8192), expected)

    def test_fp8_cache_writer_preserves_strides_padding_and_scale(self):
        for scale in [0.5, 1.0, 2.0]:
            with self.subTest(scale=scale):
                backing = torch.zeros(
                    4, 64, 5, 576, device="cuda", dtype=torch.float8_e4m3fn
                )
                cache = backing[:, :, 3, :]
                latent = torch.randn(9, 512, device="cuda", dtype=torch.bfloat16)
                rotary = torch.randn(9, 64, device="cuda", dtype=torch.bfloat16)
                slots = torch.tensor([0, 1, 63, 64, 70, 127, 128, -1], device="cuda")
                layer = SimpleNamespace(
                    kv_cache_dtype="fp8",
                    kv_cache=cache,
                    _k_scale=torch.tensor(scale, device="cuda", dtype=torch.float32),
                )
                DSparkMLACache.store(layer, latent, rotary, slots)
                expected = (
                    (torch.cat([latent[:7], rotary[:7]], dim=-1).float() / scale)
                    .to(torch.float8_e4m3fn)
                    .float()
                )
                actual = cache[slots[:7] // 64, slots[:7] % 64].float()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for i in [0, 1, 2, 4]:
                    self.assertEqual(backing[:, :, i, :].float().count_nonzero(), 0)
                self.assertEqual(cache[-1, -1].float().count_nonzero(), 0)

    def test_fp8_attention_matches_dequantized_reference(self):
        args = self.inputs([9000], [8])
        q, cache, table, starts, lengths = args
        cache = cache.to(torch.float8_e4m3fn)
        for scale in [0.5, 1.0, 2.0]:
            with self.subTest(scale=scale):
                expected = self.dense(
                    (q, cache.float() * scale, table, starts, lengths),
                    [9000],
                    [8],
                    8192,
                )
                actual = attention(
                    q,
                    cache,
                    table,
                    starts,
                    lengths,
                    True,
                    8192,
                    kv_scale=torch.tensor(scale, device="cuda", dtype=torch.float32),
                )
                self.assert_reference(actual, expected)

    def test_released_prefix_pages_are_not_read(self):
        args = self.inputs([16407], [8])
        first_visible = 16407 - 8 + 1 - 8192
        args[2][:, : first_visible // 64] = -1
        expected = self.dense(args, [16407], [8], 8192)
        self.assert_reference(attention(*args, True, 8192), expected)

    def test_actual_token_count_excludes_nan_padding(self):
        q, *rest = self.inputs([9000], [8])
        q = torch.cat([q, torch.full_like(q, float("nan"))])
        actual = attention(q, *rest, True, 8192, num_actual_tokens=8)
        expected = self.dense((q, *rest), [9000], [8], 8192)
        self.assert_reference(actual[:8], expected)
        torch.testing.assert_close(actual[8:], torch.zeros_like(actual[8:]))

    def test_graph_replay_handles_shrinking_and_empty_batches(self):
        q, cache, table, starts, lens = self.inputs([9000, 16407], [8, 8])
        q = torch.cat([q, torch.zeros_like(q)])

        def forward():
            return attention(q, cache, table, starts, lens, True, 8192)

        for _ in range(3):
            forward()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = forward()
        for live in [2, 1, 0, 2]:
            with self.subTest(live=live):
                q.normal_()
                q[live * 8 :] = float("nan")
                starts.copy_(
                    torch.tensor(
                        [0, 8 if live else 0, live * 8],
                        device="cuda",
                        dtype=starts.dtype,
                    )
                )
                lens.copy_(
                    torch.tensor(
                        [9000 if live else 0, 16407 if live > 1 else 0],
                        device="cuda",
                        dtype=lens.dtype,
                    )
                )
                graph.replay()
                torch.testing.assert_close(actual, forward(), rtol=0, atol=0)
                self.assertTrue(bool(actual.isfinite().all()))
                torch.testing.assert_close(
                    actual[live * 8 :], torch.zeros_like(actual[live * 8 :])
                )
                if live:
                    expected = self.dense(
                        (q, cache, table, starts, lens),
                        [9000, 16407][:live],
                        [8] * live,
                        8192,
                    )
                    self.assert_reference(actual[: live * 8], expected)

    def test_noncausal_full_attention(self):
        args = self.inputs([13], [3])
        self.assert_reference(
            attention(*args, False, 0),
            self.dense(args, [13], [3], 0, causal=False),
        )

    def test_noncausal_sliding_window_is_explicitly_rejected(self):
        with self.assertRaises(NotImplementedError):
            attention(*self.inputs([13], [3]), False, 8192)

    def test_empty_actual_batch(self):
        args = self.inputs([13], [3])
        actual = attention(*args, True, 8192, num_actual_tokens=0)
        torch.testing.assert_close(actual, torch.zeros_like(actual))

    def test_shared_kernel_preserves_mha_full_and_windowed_attention(self):
        q = torch.randn(1, 1, 128, device="cuda", dtype=torch.bfloat16)
        keys = torch.randn(1, 64, 1, 128, device="cuda", dtype=q.dtype)
        values = torch.randn_like(keys)
        pages = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
        lengths = torch.tensor([13], device="cuda", dtype=torch.int32)
        tmp = torch.empty((1, 1, 4, 129), device="cuda", dtype=torch.float32)
        out = torch.empty_like(q)
        lse = torch.empty((1, 1), device="cuda", dtype=torch.float32)
        for window in [None, 1, 7]:
            with self.subTest(window=window):
                decode_attention_fwd(
                    q,
                    keys,
                    values,
                    out,
                    lse,
                    pages,
                    lengths,
                    tmp,
                    4,
                    128**-0.5,
                    page_size=64,
                    sliding_window=window,
                )
                start = max(0, 13 - window) if window else 0
                scores = q[0, 0].float() @ keys[0, start:13, 0].float().T
                expected = (scores * (128**-0.5)).softmax(-1)
                expected = expected @ values[0, start:13, 0].float()
                self.assert_reference(out[0, 0], expected.to(q.dtype))


if __name__ == "__main__":
    unittest.main()
