# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM DSpark causal-window adapter to the unmodified PR63795 MLA kernel.

Only metadata is expanded here; score, softmax, value and split-reduction
arithmetic use the existing Triton decode implementation with PR63795's window.
"""

import torch

from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd


def attention(q, z, table, starts, lengths, causal, window, num_actual_tokens=None):
    """Attend query rows to paged latent512+rope64 keys and latent512 values.

    The per-request lengths include the query block. Each causal query ends
    at its own position, including when the enclosing graph pads the batch.
    ``z`` may be a strided view of a multi-layer BLHNC allocation.
    """
    assert q.is_contiguous() and q.shape[-1] == z.shape[-1] == 576
    assert z.ndim == 3 and z.stride(-1) == 1
    if window and not causal:
        raise NotImplementedError(
            "This adapter supports causal SWA and noncausal full attention. "
            "PR63795's common-end noncausal window differs from the checkpoint's "
            "per-query window semantics."
        )
    padded_tokens, heads, _ = q.shape
    tokens = padded_tokens if num_actual_tokens is None else num_actual_tokens
    assert 0 <= tokens <= padded_tokens
    if tokens == 0:
        return torch.zeros((padded_tokens, heads, 512), device=q.device, dtype=q.dtype)
    q = q[:tokens]
    rows = torch.arange(tokens, device=q.device, dtype=starts.dtype)
    request = torch.searchsorted(starts[1:], rows, right=True)
    # A full graph retains capture-size rows when runtime batches shrink.
    # Trailing query starts can coincide and zero-length requests are padding.
    request = request.clamp(max=lengths.shape[0] - 1)
    seq_lens = lengths.index_select(0, request)
    valid = (rows < starts[-1]) & (seq_lens > 0)
    if causal:
        begin = starts.index_select(0, request)
        end = starts.index_select(0, request + 1)
        seq_lens = seq_lens - (end - begin) + (rows - begin) + 1
    seq_lens = torch.where(valid, seq_lens, 1)
    pages = table.index_select(0, request)
    pages = torch.where(valid[:, None], pages, 0)
    splits = 4 if window else 16
    tmp = torch.empty(
        (tokens, heads, splits, 513), device=q.device, dtype=torch.float32
    )
    out = torch.empty((tokens, heads, 512), device=q.device, dtype=q.dtype)
    lse = torch.empty((tokens, heads), device=q.device, dtype=torch.float32)
    scale = torch.ones((), device=q.device, dtype=torch.float32)
    keys = z.unsqueeze(-2)
    decode_attention_fwd(
        q,
        keys,
        keys[..., :512],
        out,
        lse,
        pages,
        seq_lens,
        tmp,
        splits,
        0.0625,
        page_size=z.shape[1],
        k_scale=scale,
        v_scale=scale,
        is_mla=True,
        sliding_window=window or None,
    )
    out = torch.where(valid[:, None, None], out, 0)
    if padded_tokens != tokens:
        padded = torch.zeros(
            (padded_tokens, heads, 512), device=q.device, dtype=q.dtype
        )
        padded[:tokens] = out
        return padded
    return out
