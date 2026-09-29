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


def sparse_sdpa(q, kv, idx, *, scale, v_dim, rows=64):
    """sparse_sdpa (MLA latent attention over selected rows): q [1, H, S, K_DIM], kv [1, 1, T, K_DIM] (K = V source),
    idx [1, 1, S, W] int64 with -1 (the op's 0xFFFFFFFF sentinel) for an unused slot -> [1, H, S, v_dim], float32.
    out[h, s] = softmax(q[h, s] . kv[idx[s, w]] * scale over the valid w) @ kv[idx[s, w], :v_dim]. The order of the
    ids in a row does not matter. Taken from tests/unit/test_sparse_sdpa_high_precision.py (_golden)."""
    q, kvf = q.float()[0], kv.float()[0, 0]
    nh, sq = q.shape[0], q.shape[1]
    out = torch.empty(1, nh, sq, v_dim)
    for r0 in range(0, sq, rows):
        i = idx[0, 0, r0 : r0 + rows]
        valid = i >= 0
        sel = kvf[i.clamp(min=0)]  # [r, W, K_DIM]
        s = torch.einsum("hrk,rwk->hrw", q[:, r0 : r0 + rows], sel) * scale
        s = s.masked_fill(~valid[None], float("-inf"))
        out[0, :, r0 : r0 + rows] = torch.einsum("hrw,rwv->hrv", s.softmax(-1), sel[..., :v_dim])
    return out
