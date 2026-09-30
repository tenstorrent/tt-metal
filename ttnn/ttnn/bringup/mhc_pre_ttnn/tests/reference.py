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


def mhc_pre_xing_coefficients(row, scale, base, norm_width, n, norm_eps, hc_eps, sinkhorn_iters, clamp):
    """ttnn.bringup.mhc_pre_xing, coefficients_given=False (float64): the all-reduced row (..., T, 32) =
    [mix 0..n(n+2)-1 | sum x^2 | 0..] -> hc (..., T, n(n+2)) = [pre | post | comb row-major]. The Xing4.0 math of
    models/demos/xing40_a4b_d_p/reference/xing_ref.py:hc_weights after its projection: sigmoid pre with no + eps,
    clamped comb logits, exp(L - rowmax), then sinkhorn_iters x (rows / (sum + eps), columns / (sum + eps))."""
    row = row.double()
    ng = n * (n + 2)
    mix, ss = row[..., :ng], row[..., ng : ng + 1]
    b = torch.as_tensor(base, dtype=torch.float64)
    a_pre, a_post, a_res = scale
    z = mix * torch.rsqrt(ss / norm_width + norm_eps)
    pre = torch.sigmoid(z[..., :n] * a_pre + b[:n])
    post = 2 * torch.sigmoid(z[..., n : 2 * n] * a_post + b[n : 2 * n])
    logits = (z[..., 2 * n :] * a_res + b[2 * n :]).reshape(*z.shape[:-1], n, n).clamp(clamp[0], clamp[1])
    m = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
    for _ in range(sinkhorn_iters):
        m = m / (m.sum(dim=-1, keepdim=True) + hc_eps)
        m = m / (m.sum(dim=-2, keepdim=True) + hc_eps)
    return torch.cat([pre, post, m.reshape(*m.shape[:-2], n * n)], dim=-1)


def mhc_pre_xing_collapse(hc, streams, n):
    """y = sum_i pre_i * streams[..., i*C:(i+1)*C] (float64), pre = hc[..., :n]."""
    x = streams.double()
    C = x.shape[-1] // n
    return (hc.double()[..., :n].unsqueeze(-1) * x.reshape(*x.shape[:-1], n, C)).sum(dim=-2)
