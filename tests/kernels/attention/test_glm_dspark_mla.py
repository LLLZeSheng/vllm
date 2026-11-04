# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise causal windows and live-batch changes in the MLA graph path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import vllm.envs as envs
from vllm.model_executor.models.dspark_context_kv import DSparkContextKVModel
from vllm.model_executor.models.glm_dspark_mla import (
    DSparkMLACache,
    GlmDSparkMLAModel,
    rms,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.dspark_upstream_window import attention
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd


class TestGlmDSparkMLACacheSpec(unittest.TestCase):
    def test_flashmla_cache_declares_record_and_block_stride_alignment(self):
        config = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))
        cache = SimpleNamespace(kv_cache_dtype="fp8", sliding_window=8192, num_heads=64)
        with patch.object(envs, "VLLM_GLM_DSPARK_MLA_FLASHMLA", True):
            spec = DSparkMLACache.get_kv_cache_spec(cache, config)
        self.assertEqual(spec.dtype, torch.uint8)
        self.assertEqual(spec.head_size, 656)
        self.assertEqual(spec.block_stride_alignment, 656)
        self.assertEqual(spec.sliding_window, 8192)

    def test_flashmla_rejects_non_fp8_cache(self):
        config = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))
        cache = SimpleNamespace(
            kv_cache_dtype="bfloat16", sliding_window=8192, num_heads=64
        )
        with (
            patch.object(envs, "VLLM_GLM_DSPARK_MLA_FLASHMLA", True),
            self.assertRaisesRegex(ValueError, "requires FP8"),
        ):
            DSparkMLACache.get_kv_cache_spec(cache, config)

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


class TestGlmDSparkMLAContextProjection(unittest.TestCase):
    def test_producer_and_drafter_match_separate_layer_projections(self):
        class RecordingAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.kv_a_proj_with_mqa = nn.Linear(4, 576, bias=False)
                self.written = None

            def store_projected(self, projected, positions, slots):
                if slots is not None:
                    self.written = projected.clone(), positions, slots

        torch.manual_seed(7)
        states = torch.randn(3, 4)
        positions = torch.arange(3)
        slots = [torch.arange(3), None, torch.arange(3), None, torch.arange(3)]
        for model_cls in (DSparkContextKVModel, GlmDSparkMLAModel):
            with self.subTest(model=model_cls.__name__):
                model = nn.Module()
                model.layers = nn.ModuleList()
                for _ in range(5):
                    layer = nn.Module()
                    layer.self_attn = RecordingAttention()
                    model.layers.append(layer)
                model.hidden_norm = nn.Module()
                model.hidden_norm.register_buffer("weight", torch.randn(4))
                model.config = SimpleNamespace(rms_norm_eps=1e-5)
                model.is_mla = True

                normalized = rms(states, model.hidden_norm.weight, 1e-5)
                expected = [
                    layer.self_attn.kv_a_proj_with_mqa(normalized)
                    for layer in model.layers
                ]
                model_cls._build_fused_kv_buffers(model)
                self.assertEqual(model._fused_mla_context_kv_weight.shape, (2880, 4))
                self.assertFalse(model._fused_mla_context_kv_weight.requires_grad)

                original_linear = F.linear
                with patch(
                    "torch.nn.functional.linear", wraps=original_linear
                ) as linear:
                    model_cls.precompute_and_store_context_kv(
                        model, states, positions, slots
                    )
                self.assertEqual(linear.call_count, 1)
                for i, layer in enumerate(model.layers):
                    if slots[i] is None:
                        self.assertIsNone(layer.self_attn.written)
                    else:
                        written, written_positions, written_slots = (
                            layer.self_attn.written
                        )
                        torch.testing.assert_close(written, expected[i])
                        self.assertIs(written_positions, positions)
                        self.assertIs(written_slots, slots[i])


class TestGlmDSparkMLAReuseScope(unittest.TestCase):
    def test_reuse_isolated_across_nested_calls_and_exceptions(self):
        from vllm.v1.attention.ops.glm_dspark_mla import REUSE, forward_scope

        @forward_scope
        def nested(fail=False):
            self.assertEqual(REUSE.get(), {})
            REUSE.get()["visited"] = True
            if fail:
                raise RuntimeError("abort forward")

        @forward_scope
        def outer():
            reuse = REUSE.get()
            reuse["outer"] = True
            nested()
            self.assertIs(REUSE.get(), reuse)
            with self.assertRaisesRegex(RuntimeError, "abort forward"):
                nested(fail=True)
            self.assertEqual(REUSE.get(), {"outer": True})

        with patch.object(envs, "VLLM_GLM_DSPARK_MLA_OPTIMIZATIONS", "planner"):
            outer()
            outer()
        self.assertIsNone(REUSE.get())


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


@unittest.skipUnless(current_platform.is_cuda(), "CUDA is required")
class TestGlmDSparkFlashMLA(unittest.TestCase):
    def test_fused_rope_preserves_finite_bf16_patterns_including_subnormals(self):
        from vllm.model_executor.models.glm_dspark_mla import GlmDSparkMLAAttention
        from vllm.v1.attention.backends.mla import dspark_flashmla
        from vllm.v1.attention.ops import glm_dspark_mla as ops

        if not current_platform.has_device_capability(90):
            self.skipTest("Packed FP8 cache requires a FlashMLA-capable GPU")
        bits = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16)
        values = bits.view(torch.bfloat16)
        values = torch.where(values.isfinite(), values, 0)
        torch.manual_seed(48)
        positions = torch.randint(0, 1048576, (1024,), device="cuda")
        for neox in (False, True):
            layer = SimpleNamespace(rope_theta=8000000, is_neox_style=neox)
            for _ in range(4):
                x = values[torch.randperm(len(values), device="cuda")].view(1024, 1, 64)
                with patch.object(envs, "VLLM_GLM_DSPARK_MLA_OPTIMIZATIONS", ""):
                    expected = GlmDSparkMLAAttention.rope(layer, x, positions)
                with (
                    patch.object(envs, "VLLM_GLM_DSPARK_MLA_FLASHMLA", True),
                    patch.object(envs, "VLLM_GLM_DSPARK_MLA_OPTIMIZATIONS", "rope"),
                ):
                    actual = ops.rope(layer, x, positions)
                self.assertTrue(
                    torch.equal(expected.view(torch.int16), actual.view(torch.int16))
                )
                z = torch.randn(1024, 576, dtype=torch.bfloat16, device="cuda")
                z[:, 512:] = x.squeeze(1)
                slots = torch.arange(1024, device="cuda")
                slots[-1] = -1
                layer.kv_a_layernorm = lambda x: x
                for scale_value in (0.5, 3.0):
                    backing = torch.full(
                        (16, 5, 64, 656), 91, dtype=torch.uint8, device="cuda"
                    )
                    cache = backing[:, 2]
                    scale = torch.tensor(scale_value, device="cuda")
                    layer.attn = SimpleNamespace(kv_cache=cache, _k_scale=scale)
                    dspark_flashmla.store(
                        z[:, :512], expected.squeeze(1), cache, slots, scale
                    )
                    expected_bytes = backing.clone()
                    backing.fill_(91)
                    ops.store_projected(layer, z, positions, slots)
                    self.assertTrue(torch.equal(expected_bytes, backing))

    def test_fused_rope_and_store_are_bitwise_equal_with_strides_and_padding(self):
        from vllm.model_executor.models.glm_dspark_mla import GlmDSparkMLAAttention
        from vllm.v1.attention.backends.mla import dspark_flashmla
        from vllm.v1.attention.ops import glm_dspark_mla as ops

        if not current_platform.has_device_capability(90):
            self.skipTest("Packed FP8 cache requires a FlashMLA-capable GPU")
        torch.manual_seed(47)
        positions = torch.tensor(
            [0, 1, 63, 64, 8191, 8192, 8193, 80000, 1048575], device="cuda"
        )
        slots = torch.tensor([0, 1, 63, 64, 70, 127, 128, -1], device="cuda")
        for neox in (False, True):
            layer = SimpleNamespace(
                rope_theta=8000000,
                is_neox_style=neox,
                kv_a_layernorm=lambda x: x,
            )
            x = torch.randn(9, 64, 256, dtype=torch.bfloat16, device="cuda")[..., 192:]
            with patch.object(envs, "VLLM_GLM_DSPARK_MLA_OPTIMIZATIONS", ""):
                expected = GlmDSparkMLAAttention.rope(layer, x, positions)
            with (
                patch.object(envs, "VLLM_GLM_DSPARK_MLA_FLASHMLA", True),
                patch.object(envs, "VLLM_GLM_DSPARK_MLA_OPTIMIZATIONS", "rope"),
            ):
                actual = ops.rope(layer, x, positions)
            self.assertTrue(
                torch.equal(expected.view(torch.int16), actual.view(torch.int16))
            )
            z = torch.randn(9, 5, 576, dtype=torch.bfloat16, device="cuda")[:, 3]
            with patch.object(envs, "VLLM_GLM_DSPARK_MLA_OPTIMIZATIONS", ""):
                rotary = GlmDSparkMLAAttention.rope(
                    layer, z[:, 512:].unsqueeze(1), positions
                )
            for scale_value in (0.125, 1.0, 3.0):
                backing = torch.full(
                    (4, 5, 64, 656), 91, dtype=torch.uint8, device="cuda"
                )
                cache = backing[:, 2]
                scale = torch.tensor(scale_value, device="cuda")
                layer.attn = SimpleNamespace(kv_cache=cache, _k_scale=scale)
                dspark_flashmla.store(
                    z[:, :512], rotary.squeeze(1), cache, slots, scale
                )
                expected_bytes = backing.clone()
                backing.fill_(91)
                ops.store_projected(layer, z, positions, slots)
                self.assertTrue(torch.equal(expected_bytes, backing))

    def test_packed_fp8_preserves_values_causal_windows_and_live_graph_rows(self):
        from vllm.v1.attention.backends.mla import dspark_flashmla
        from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

        if not is_flashmla_sparse_supported()[0]:
            self.skipTest(
                "FlashMLA sparse requires a supported GPU and native extension"
            )
        torch.manual_seed(37)
        reference = TestGlmDSparkMLA()
        q, _, table, starts, lengths = reference.inputs([63, 8200], [8, 8])
        pages = int(table.max()) + 1
        stride = ((3052800 + 655) // 656) * 656
        backing = torch.zeros(pages * stride, device="cuda", dtype=torch.uint8)
        cache = backing.as_strided((pages, 64, 656), (stride, 656, 1))
        latent = torch.randn(pages * 64, 512, device="cuda", dtype=q.dtype)
        rope = torch.randn(pages * 64, 64, device="cuda", dtype=q.dtype)
        slots = torch.arange(pages * 64, device="cuda", dtype=torch.int64)
        scale = torch.tensor(0.5, device="cuda", dtype=torch.float32)
        dspark_flashmla.store(latent, rope, cache, slots, scale)
        original = torch.cat([latent, rope], -1).float().div(scale)
        original = original.clamp(-448, 448).to(torch.float8_e4m3fn)
        restored_latent = cache[..., :512].contiguous().view(torch.float8_e4m3fn)
        torch.testing.assert_close(
            restored_latent.view(torch.uint8),
            original[:, :512].contiguous().view(torch.uint8).view_as(restored_latent),
            rtol=0,
            atol=0,
        )
        restored_rope = cache[..., 528:].contiguous().view(torch.bfloat16)
        expected = (original.float() * scale).view(pages, 64, 576)
        torch.testing.assert_close(restored_rope, expected[..., 512:].to(q.dtype))
        before = cache[0, 0].clone()
        dspark_flashmla.store(latent[:1], rope[:1], cache, slots[:1] - 1, scale)
        torch.testing.assert_close(cache[0, 0], before, rtol=0, atol=0)

        from vllm.v1.attention.ops.glm_dspark_mla import forward_scope

        @forward_scope
        def forward():
            first = dspark_flashmla.attention(
                q, cache, table, starts, lengths, True, 8192, kv_scale=scale
            )
            # A later layer must safely reuse the first layer's live schedule.
            return (
                dspark_flashmla.attention(
                    q, cache, table, starts, lengths, True, 8192, kv_scale=scale
                )
                + first * 0
            )

        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = forward()
        for live_lengths in ([63, 8200], [64, 8199]):
            lengths.copy_(torch.tensor(live_lengths, device="cuda", dtype=torch.int32))
            graph.replay()
            dense = reference.dense(
                (q, expected, table, starts, lengths), live_lengths, [8, 8], 8192
            )
            relative = (actual.float() - dense.float()).norm() / dense.float().norm()
            self.assertLess(float(relative), 0.01)
        starts.copy_(torch.tensor([0, 8, 8], device="cuda", dtype=torch.int32))
        lengths[1] = 0
        graph.replay()
        self.assertTrue(bool(actual.isfinite().all()))
        torch.testing.assert_close(actual[8:], torch.zeros_like(actual[8:]))


if __name__ == "__main__":
    unittest.main()
