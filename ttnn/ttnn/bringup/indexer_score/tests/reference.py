# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Torch semantics of the indexer_score fork's DSA scorer (ttnn.bringup.indexer_score_dsa /
ring_indexer_score_dsa), for the fork's model-case tests.

    score[s, t] = sum_h relu(q[h, s, :] . k[t, :]) * w[s, h]    for t <= pos(s), else -inf

(the op's docstring; the source op's tests/ttnn/nightly/.../indexer_score/test_indexer_score.py:indexer_score_dsa_ref).
The fork's change (fp32 DEST, opt-in) only makes the MAC more accurate; the semantics are the source op's.

Ring layout with block_cyclic_cache_tp_sharded on an SP x TP mesh (the op's docstring and the source op's
ring_indexer_score_test_utils._to_tp_inner_reconstructed): the key cache is striped over the sp * tp chips, chip
L = sp_rank * tp + tp_rank holding, for every slab j of chunk = sp * block_cyclic_chunk_local keys, the positions
j * chunk + L * (chunk / (sp * tp)) + [0, chunk / (sp * tp)). k_local on SP rank r is the TP-inner gather of its row's
stripes (tp_rank 0's stripe, then tp_rank 1's, ...). Query rows: device (sp_rank, tp_rank) holds the rows at
chunk_start_idx + sp_rank * chunk_local + tp_rank * Sq + [0, Sq), Sq = q's rows, chunk_local =
block_cyclic_chunk_local.
"""

from __future__ import annotations

import torch


def k_local_positions(t: int, sp: int, tp: int, chunk_local: int) -> torch.Tensor:
    """[sp, t / sp] int64: natural key position of each row of SP rank r's k_local (TP-inner gather order)."""
    chunk = sp * chunk_local
    stripe_chunk = chunk // (sp * tp)
    stripe_rows = t // (sp * tp)
    assert t % chunk == 0 and chunk % (sp * tp) == 0
    i = torch.arange(stripe_rows)
    j, w = i // stripe_chunk, i % stripe_chunk
    out = torch.empty(sp, t // sp, dtype=torch.int64)
    for r in range(sp):
        for c in range(tp):
            chip = r * tp + c
            out[r, c * stripe_rows : (c + 1) * stripe_rows] = j * chunk + chip * stripe_chunk + w
    assert torch.equal(out.flatten().sort().values, torch.arange(t))
    return out


def query_start(chunk_start: int, sp_rank: int, tp_rank: int, rows: int, chunk_local: int) -> int:
    """Global position of device (sp_rank, tp_rank)'s first query row."""
    return chunk_start + sp_rank * chunk_local + tp_rank * rows


def dsa_score(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, q_start: int, key_stride: int = 1) -> torch.Tensor:
    """q [H, S, D], k [T, D] (natural order), w [S, H] -> [S, T] float32; key t visible to row s iff
    t <= q_start + s, the rest -inf.

    key_stride R (the fork's pooled-key mode): key t pools query tokens [R*t, R*t + R) and is visible to row s iff
    R*t + R - 1 <= q_start + s (q_start in tokens, t in keys). R = 1 is the plain causal mask above."""
    q, k, w = q.float(), k.float(), w.float()
    kt = k.t().contiguous()
    score = torch.zeros(q.shape[1], k.shape[0])
    for h in range(q.shape[0]):
        score.add_(torch.relu(q[h] @ kt).mul_(w[:, h : h + 1]))
    r = key_stride
    future = (r * torch.arange(k.shape[0])[None, :] + r - 1) > (q_start + torch.arange(q.shape[1]))[:, None]
    return score.masked_fill_(future, float("-inf"))


def full_mesh_key_local_positions(t: int, n_dev: int, key_stripe: int) -> torch.Tensor:
    """[n_dev, t / n_dev] int64: natural key of each k_local row on a full-mesh block-cyclic ring (key units).
    Chip d (row-major tensor rank) holds, for every chunk c of n_dev * key_stripe keys, keys
    c * n_dev * key_stripe + d * key_stripe + [0, key_stripe) at local rows c * key_stripe + [0, key_stripe)."""
    chunk = n_dev * key_stripe
    assert t % chunk == 0
    i = torch.arange(t // n_dev)
    c, w = i // key_stripe, i % key_stripe
    out = torch.stack([c * chunk + d * key_stripe + w for d in range(n_dev)])
    assert torch.equal(out.flatten().sort().values, torch.arange(t))
    return out


# FPU operand mantissa bits (in0 = srcB, in1 = srcA) per fidelity on Blackhole, truncated (deepseek_v41_flash_d_p O.1,
# probe mhc_post_ds41 / DS41-108; the in1 bits are DS41-42's): the formats the scorer's two FPU steps multiply in.
FPU_BITS = {"LoFi": (6, 4), "HiFi2": (6, 9), "HiFi3": (10, 9), "HiFi4": (10, 9)}


def _truncate(t, bits):
    i = t.float().contiguous().view(torch.int32)
    sh = 23 - bits
    return ((i >> sh) << sh).view(torch.float32)


def dsa_score_at(q, k, w, q_start, key_stride=1, fidelity=None):
    """dsa_score in q's dtype (float64: the reference; float32: noise_limit's formats run) -> the visible scores
    [S, T] with the future columns -inf. ``fidelity`` (float32 only): the FPU's operands as the scorer multiplies them,
    q.k^T (q in0, k in1) then relu(qk) x w (qk in1 = srcA, the gate w in0 = srcB, broadcast over the columns), every
    product and sum in fp32 (compute_indexer_score.cpp: matmul_block, then mul_tiles_bcast_cols MAC'd in DEST)."""
    dt = q.dtype
    fpu = fidelity is not None and dt == torch.float32
    b0, b1 = FPU_BITS[fidelity] if fpu else (None, None)
    kt = (_truncate(k, b1) if fpu else k.to(dt)).t().contiguous()
    w = _truncate(w, b0) if fpu else w.to(dt)
    score = torch.zeros(q.shape[1], k.shape[0], dtype=dt)
    for h in range(q.shape[0]):
        qh = _truncate(q[h], b0) if fpu else q[h].to(dt)
        qk = torch.relu(qh @ kt)
        if fpu:
            qk = _truncate(qk, b1)
        score.add_(qk.mul_(w[:, h : h + 1]))
    r = key_stride
    future = (r * torch.arange(k.shape[0])[None, :] + r - 1) > (q_start + torch.arange(q.shape[1]))[:, None]
    return score.masked_fill_(future, float("-inf"))


def bfp_grid(x: torch.Tensor, mbits: int = 3) -> torch.Tensor:
    """x on tt-metal's host bfloat4_b (mbits 3) / bfloat8_b (7) grid (convert_u32_to_bfp: per 16 values along the last
    dim the max exponent, each significand shifted to it, rounded to nearest even, clamped), as fp32: inputs that
    ttnn.from_torch packs exactly."""
    u = x.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exp, sign, man = (u >> 23) & 0xFF, u >> 31, (u & 0x7FFFFF) | (1 << 23)
    eg = exp.view(-1, 16)
    e = eg.amax(-1, keepdim=True)
    m = man.view(-1, 16) >> (e - eg).clamp(max=62)
    sh = 24 - mbits
    q, rv, half = m >> sh, m & ((1 << sh) - 1), 1 << (sh - 1)
    q = (q + ((rv > half) | ((rv == half) & ((q & 1) == 1))).long()).clamp(max=(1 << mbits) - 1)
    q = torch.where(eg == 0, torch.zeros_like(q), q)
    v = q.double() * torch.pow(2.0, (e - 127 - (mbits - 1)).double())
    return torch.where(sign.view(-1, 16) == 1, -v, v).float().view(x.shape)
