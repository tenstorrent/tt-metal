# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_pre in torch (float64), from the op's docstring and op_design.md (DeepSeek-V4 mHC order)."""

import torch


def mhc_pre(x, w, b, scale, sinkhorn_iters, eps, norm_eps):
    """x (..., T, n*C), w (n*C, n (n + 2)), b (1, n (n + 2)) -> (y (..., T, C), post (..., T, n), comb (..., T, n*n))."""
    x, w, b = x.double(), w.double(), b.double().reshape(-1)
    mix_w = w.shape[-1]
    n = next(k for k in range(1, 64) if k * (k + 2) == mix_w)
    a_pre, a_post, a_res = scale
    r = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mix = (x @ w) * r
    pre = torch.sigmoid(a_pre * mix[..., :n] + b[:n]) + eps
    post = 2 * torch.sigmoid(a_post * mix[..., n : 2 * n] + b[n : 2 * n])
    logits = (a_res * mix[..., 2 * n :] + b[2 * n :]).reshape(*mix.shape[:-1], n, n)
    m = torch.softmax(logits, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    C = x.shape[-1] // n
    y = (pre.unsqueeze(-1) * x.reshape(*x.shape[:-1], n, C)).sum(dim=-2)
    return y, post, m.reshape(*m.shape[:-2], n * n)
