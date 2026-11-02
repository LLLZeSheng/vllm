# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise causal windows and live-batch changes in the MLA graph path."""

import unittest

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.dspark_upstream_window import attention
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd


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
