# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The sdpa fork's ops in torch (fp32), for the model cases in cases.py.

ring_mla: causal latent-V attention. K and V are one shared KV head; QK^T uses all of KV's columns, V is its first
head_dim_v columns. The fork's changes (fp32 DEST on the latent-V streaming path, exact_score_exp) only make the
device closer to exact softmax attention, so the reference is the exact attention.

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
