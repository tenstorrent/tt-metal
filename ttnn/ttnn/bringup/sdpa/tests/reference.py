# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of the sdpa fork's scaled_dot_product_attention / chunked_scaled_dot_product_attention, including
the fork's change: V may be narrower than Q/K (non-MLA GQA), and the output then has V's width.

out[h] = softmax(Q[h] K[g]^T * scale + mask [+ sink column]) V[g], g = h // (NQH // NKV), float32, computed in
blocks of query rows. Q holds positions [q_start, q_start + Sq); causal: key j visible to query i iff j <= i; with a
sliding window W also j > i - W. attention_sink[h] is unscaled, like the op's: its logit is sink[h] * scale, it joins
the softmax denominator only (no V row). Taken from the fork's unit tests (tests/unit/test_sdpa_narrow_v.py) and the
op's docstring, not from any model output."""

import torch


def sdpa(q, k, v, *, scale, q_start=0, window=0, sink=None, rows=1024):
    """q [1, NQH, Sq, DK], k [1, NKV, Sk, DK], v [1, NKV, Sk, DV] (contiguous logical order) -> [1, NQH, Sq, DV]."""
    q, k, v = q.float(), k.float(), v.float()
    nqh, sq = q.shape[1], q.shape[2]
    nkv, sk = k.shape[1], k.shape[2]
    out = torch.empty(1, nqh, sq, v.shape[-1])
    j = torch.arange(sk)[None, :]
    for r0 in range(0, sq, rows):
        r1 = min(sq, r0 + rows)
        i = torch.arange(q_start + r0, q_start + r1)[:, None]
        visible = j <= i
        if window:
            visible = visible & (j > i - window)
        for h in range(nqh):
            g = h // (nqh // nkv)
            s = (q[0, h, r0:r1] @ k[0, g].T) * scale
            s = s.masked_fill(~visible, float("-inf"))
            if sink is not None:
                s = torch.cat([s, torch.full((r1 - r0, 1), float(sink[h]) * scale)], dim=-1)
            p = torch.softmax(s, dim=-1)[:, :sk]
            out[0, h, r0:r1] = p @ v[0, g]
    return out


def unpage(paged, page_table):
    """Paged cache [nb_phys, NKV, B, D] and a page table [nblocks] (logical block b at physical page_table[b]) ->
    contiguous [1, NKV, nblocks * B, D]."""
    blocks = paged[page_table.long()]  # [nblocks, NKV, B, D]
    nb, nkv, b, d = blocks.shape
    return blocks.transpose(0, 1).reshape(1, nkv, nb * b, d)
