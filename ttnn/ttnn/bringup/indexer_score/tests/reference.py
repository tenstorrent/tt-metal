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


def dsa_score(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, q_start: int) -> torch.Tensor:
    """q [H, S, D], k [T, D] (natural order), w [S, H] -> [S, T] float32; key t visible to row s iff
    t <= q_start + s, the rest -inf."""
    q, k, w = q.float(), k.float(), w.float()
    kt = k.t().contiguous()
    score = torch.zeros(q.shape[1], k.shape[0])
    for h in range(q.shape[0]):
        score.add_(torch.relu(q[h] @ kt).mul_(w[:, h : h + 1]))
    future = torch.arange(k.shape[0])[None, :] > (q_start + torch.arange(q.shape[1]))[:, None]
    return score.masked_fill_(future, float("-inf"))
