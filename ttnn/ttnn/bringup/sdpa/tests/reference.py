# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The sdpa fork's ops in torch (fp32), for the model cases in cases.py.

ring_mla: causal latent-V attention. K and V are one shared KV head; QK^T uses all of KV's columns, V is its first
head_dim_v columns. The fork's changes (fp32 DEST on the latent-V streaming path, exact_score_exp) only make the
device closer to exact softmax attention, so the reference is the exact attention.

sparse_sdpa: absorbed MQA over the top-k latents named by `indices` (one shared KV head; V = the leading v_dim columns
of the K_DIM-wide kvpe). An index of MASKED_INDEX (0xFFFFFFFF) scores -inf; with an attention sink, each head's sink
logit is `sink * scale` and joins the softmax denominator only. The fork's exact_exp changes the precision of the
score exp, not the result, so it has no term here.

torch is imported inside the functions: ttnn's pre-commit forbids a global torch import under ttnn/ttnn.
"""


def block_cyclic_rows(n: int, sp: int, chunk_local: int):
    """For global positions 0..n-1: the row of the SP-concatenated cache (device-major, device d's local rows after
    d * n / sp) that holds each one. Position g is on device (g // chunk_local) % sp at local row
    (g // (sp * chunk_local)) * chunk_local + g % chunk_local: the chunked cache layout ring_mla reads with
    kv_actual_isl."""
    import torch

    g = torch.arange(n)
    dev = (g // chunk_local) % sp
    local = (g // (sp * chunk_local)) * chunk_local + g % chunk_local
    return dev * (n // sp) + local


def ring_mla(q, kv, *, head_dim_v: int, scale: float, q_start: int, logical_n: int, heads_per_step: int = 4):
    """q [1, H, Sq, D] at absolute positions q_start .. q_start + Sq - 1; kv [1, 1, N, D] in position order.
    Query at position p attends to keys j <= p, j < logical_n. Returns [1, H, Sq, head_dim_v] fp32."""
    import torch

    _, nh, sq, _ = q.shape
    k = kv[0, 0, :logical_n].float()
    v = k[:, :head_dim_v]
    pos = torch.arange(q_start, q_start + sq)
    masked = torch.arange(k.shape[0])[None, :] > pos[:, None]  # [Sq, N]
    out = torch.empty(1, nh, sq, head_dim_v, dtype=torch.float32)
    for h0 in range(0, nh, heads_per_step):
        s = (q[0, h0 : h0 + heads_per_step].float() @ k.T) * scale  # [h, Sq, N]
        s.masked_fill_(masked, float("-inf"))
        out[0, h0 : h0 + heads_per_step] = torch.softmax(s, dim=-1) @ v
    return out


MASKED_INDEX = 0xFFFFFFFF


def sparse_sdpa(q, kv, indices, scale, v_dim, attention_sink=None, rows=16):
    """q [1,H,S,K], kv [1,1,T,K], indices [1,1,S,k] int64, attention_sink [1,1,1,H] -> [1,H,S,v_dim] fp32.
    Computed `rows` query rows at a time (the gathered keys are [rows, k, K])."""
    import torch

    q, kv = q.double()[0], kv.double()[0, 0]  # [H,S,K], [T,K]
    idx = indices[0, 0].to(torch.int64)  # [S,k]
    H, S, _ = q.shape
    out = torch.empty(H, S, v_dim, dtype=torch.float64)
    sink = None if attention_sink is None else attention_sink.double().reshape(H, 1, 1) * scale
    for s0 in range(0, S, rows):
        ix = idx[s0 : s0 + rows]
        masked = ix == MASKED_INDEX
        sel = kv[torch.where(masked, torch.zeros_like(ix), ix)]  # [r,k,K]
        scores = torch.einsum("hsd,sjd->hsj", q[:, s0 : s0 + rows], sel) * scale
        scores = scores.masked_fill(masked.unsqueeze(0), float("-inf"))
        if sink is not None:
            scores = torch.cat([scores, sink.expand(H, scores.shape[1], 1)], dim=-1)
        probs = scores.softmax(dim=-1)
        if sink is not None:
            probs = probs[..., :-1]
        out[:, s0 : s0 + rows] = torch.einsum("hsj,sjd->hsd", probs, sel[..., :v_dim])
    return out.unsqueeze(0).float()
