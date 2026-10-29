# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""torch semantics of the sdpa fork's ops, as the model cases (cases.py) call them.

ring_joint_scaled_dot_product_attention, metadata chunked path (slot_id / kv_actual_isl_tensor): the chunk's queries
are the global positions [kv_actual, kv_actual + chunk); each query at position p attends causally to the cache keys
of the selected slot at positions <= p (sliding_window_size w: positions > p - w), keys past kv_actual + chunk are never
read. attention_sink adds one logit per query head, sink * scale, to the softmax denominator only (the source op's
golden, ttnn/ttnn/operations/transformer_golden.py `_scaled_attention`). exact_exp only changes how exp is evaluated,
so the fork's change has the same reference. With a window, the persistent gather buffers hold, per chip and per run of
consecutive query rows (at most two: a chunk that wraps an SP slab), the window - 1 keys before the run rounded up to
whole k chunks (the compact predecessor halo).
"""

from __future__ import annotations


def chunk_positions(start: int, sp: int, chunk: int) -> torch.Tensor:
    """[sp, chunk // sp]: the global position of each q row of SP row r for a chunk at ``start`` (any 32-aligned
    start): the chunk's positions g with (g // W) % sp == r, in increasing order (the server's block-cyclic
    reshuffle; the op's KV-pad rotation)."""
    import torch

    w = chunk // sp
    g = torch.arange(start, start + chunk)
    rows = (g // w) % sp
    return torch.stack([g[rows == r] for r in range(sp)])


def cache_local_index(sp: int, chunk: int, local_seq: int) -> torch.Tensor:
    """[sp, local_seq]: the global position held by local row L of SP row r: (L // W) * chunk + r * W + L % W."""
    import torch

    w = chunk // sp
    L = torch.arange(local_seq)
    return torch.stack([(L // w) * chunk + r * w + L % w for r in range(sp)])


def halo_positions(p: torch.Tensor, halo: int, w: int, slots: int = 2) -> torch.Tensor:
    """[slots * halo]: key positions of the halo slots of one chip's q rows p (W = w rows per slab). Slot s holds the
    last ``halo`` rows of the slab before the s-th run's first slab (kernels/sliding_window_work_plan.hpp
    sliding_halo_sources), so a run that starts mid-slab reads the rows between its slab's start and its first row
    from the chip's own slab, not from the halo. An unused slot or a position < 0 is -1."""
    import torch

    breaks = [0] + [i for i in range(1, len(p)) if p[i] != p[i - 1] + 1]
    out = torch.full((slots, halo), -1, dtype=torch.int64)
    for s, b in enumerate(breaks[:slots]):
        out[s] = (int(p[b]) // w) * w - halo + torch.arange(halo)
    out[out < 0] = -1
    return out.flatten()


def ring_sdpa(q, k, v, q_pos, scale, window=None, sink=None, block=256):
    """q [H, P, D] at global positions q_pos [P]; k [KV, S, D], v [KV, S, Dv] in position order (S > max(q_pos));
    sink [H] or None -> [H, P, Dv] float32. GQA: q head h reads kv head h // (H // KV)."""
    import torch

    H, P, _ = q.shape
    KV, Dv = k.shape[0], v.shape[-1]
    rep = H // KV
    out = torch.empty(H, P, Dv)
    for b0 in range(0, P, block):
        qp = q_pos[b0 : b0 + block]
        lo = max(0, int(qp.min()) - window + 1) if window else 0
        hi = int(qp.max()) + 1
        kp = torch.arange(lo, hi)
        ok = kp[None, :] <= qp[:, None]
        if window:
            ok &= kp[None, :] > qp[:, None] - window
        for g in range(KV):
            qg = q[g * rep : (g + 1) * rep, b0 : b0 + block].float()
            s = (qg @ k[g, lo:hi].float().T) * scale
            s = s.masked_fill(~ok, float("-inf"))
            if sink is not None:
                sk = (sink[g * rep : (g + 1) * rep].float() * scale).view(rep, 1, 1).expand(rep, s.shape[1], 1)
                p = torch.softmax(torch.cat((s, sk), dim=-1), dim=-1)[..., :-1]
            else:
                p = torch.softmax(s, dim=-1)
            out[g * rep : (g + 1) * rep, b0 : b0 + block] = p @ v[g, lo:hi].float()
    return out
