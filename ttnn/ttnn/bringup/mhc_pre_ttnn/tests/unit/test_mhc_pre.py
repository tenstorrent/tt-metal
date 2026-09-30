# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for mhc_pre (immutable spec — the implementer must not edit).

mhc_pre(X, W, b, scale=(a_pre, a_post, a_res)) -> (y, post, comb), per token row x (length n*C):
    r     = rsqrt(mean(x^2) + norm_eps)
    mixes = (x @ W) * r
    pre   = sigmoid(a_pre  * mixes[0:n]  + b[0:n]) + eps
    post  = 2 * sigmoid(a_post * mixes[n:2n] + b[n:2n])
    comb  = Sinkhorn(a_res * mixes[2n:] + b[2n:])     (n x n, DeepSeek-V4 order)
    y     = sum_i pre[i] * x[i*C:(i+1)*C]

Phase-0 cell: float32 streams, float32 (tf32-representable) weight, TILE, fp32 DEST.
Shapes cover the op's blocking branches on an 11x10 grid: single-core group (C=32),
multi-row groups (small T), multi-block per core (T=640, C=7168), >1 token tile-row
per block (C=1792), ragged T, and rank-4 batch > 1 (per-image tile padding).
"""

import pytest
import torch
import ttnn

from ttnn.bringup.mhc_pre_ttnn import mhc_pre, default_compute_kernel_config
from ttnn.operations._op_contract import UnsupportedAxisValue

N_HC = 4
MIX = N_HC * (N_HC + 2)

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}


def round_to_tf32(t):
    b = t.to(torch.float32).contiguous().view(torch.int32)
    drop = 13
    b = b + ((1 << (drop - 1)) - 1) + ((b >> drop) & 1)
    return (b & ~((1 << drop) - 1)).view(torch.float32)


def sinkhorn_ref(logits, iters, eps):
    m = torch.softmax(logits, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


def torch_mhc_pre(x, w, b, scale, iters=20, eps=1e-6, norm_eps=1e-6):
    n = N_HC
    lead, nc = tuple(x.shape[:-1]), x.shape[-1]
    C = nc // n
    xf = x.to(torch.float64).reshape(-1, nc)
    wf = w.to(torch.float64)
    bf = b.to(torch.float64).reshape(-1)
    a_pre, a_post, a_res = scale
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    mixes = (xf @ wf) * r
    pre = torch.sigmoid(mixes[:, :n] * a_pre + bf[:n]) + eps
    post = 2.0 * torch.sigmoid(mixes[:, n : 2 * n] * a_post + bf[n : 2 * n])
    logits = (mixes[:, 2 * n :] * a_res + bf[2 * n :]).reshape(-1, n, n)
    comb = sinkhorn_ref(logits, iters, eps).reshape(-1, n * n)
    y = (pre.unsqueeze(-1) * xf.reshape(-1, n, C)).sum(dim=1)
    return y.reshape(*lead, C), post.reshape(*lead, n), comb.reshape(*lead, n * n)


def pcc(a, b):
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    if den == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float((a @ b) / den)


def make_inputs(x_shape, seed=42, logit_scale=1.0):
    torch.manual_seed(seed)
    nc = x_shape[-1]
    x = torch.randn(x_shape, dtype=torch.float32)
    w = round_to_tf32(torch.randn((nc, MIX), dtype=torch.float32) / nc**0.5)
    b = torch.randn((1, MIX), dtype=torch.float32)
    scale = (1.0, 1.0, float(logit_scale))
    return x, w, b, scale


def to_dev(t, device, dtype):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


SHAPES = [
    (32, 4 * 32),  # single tile of tokens, C=32 (one-core groups)
    (64, 4 * 256),  # multi-tile, C=256
    (1, 128, 4 * 1024),  # rank 3, small T -> multi-row groups
    (1, 2, 64, 4 * 512),  # rank 4, batch 2 (per-image tile padding)
    (1, 100, 4 * 256),  # ragged T
    (17, 4 * 128),  # ragged, sub-tile T
    (1, 1, 640, 4 * 1792),  # DeepSeek-V4 TP4: >1 token tile-row per block
    (1, 1, 640, 4 * 7168),  # DeepSeek-V4 full hidden: multi-block per core
]


def _check(y, post, comb, x, w, b, scale, dtype, iters=20):
    n = N_HC
    y_ref, post_ref, comb_ref = torch_mhc_pre(x, w, b, scale, iters=iters)
    lead = list(x.shape[:-1])
    C = x.shape[-1] // n

    assert list(y.shape) == lead + [C]
    assert list(post.shape) == lead + [n]
    assert list(comb.shape) == lead + [n * n]
    assert y.dtype == dtype
    assert post.dtype == ttnn.float32 and comb.dtype == ttnn.float32
    for t in (y, post, comb):
        assert t.layout == ttnn.TILE_LAYOUT

    y_t = ttnn.to_torch(y).to(torch.float64)
    post_t = ttnn.to_torch(post).to(torch.float64)
    comb_t = ttnn.to_torch(comb).to(torch.float64)
    for t in (y_t, post_t, comb_t):
        assert torch.isfinite(t).all()

    assert pcc(y_t, y_ref) >= PCC[dtype], f"y pcc {pcc(y_t, y_ref)}"
    assert pcc(post_t, post_ref) >= PCC[dtype], f"post pcc {pcc(post_t, post_ref)}"
    assert pcc(comb_t, comb_ref) >= PCC[dtype], f"comb pcc {pcc(comb_t, comb_ref)}"

    # comb must be doubly stochastic (column sums exact to fp32 noise, no bias).
    m = comb_t.reshape(-1, n, n)
    col_dev = m.sum(dim=-2) - 1
    assert (m >= 0).all()
    assert abs(col_dev.mean().item()) < 1e-5, f"mean col-sum bias {col_dev.mean().item()}"
    assert col_dev.abs().max().item() < 5e-5, f"max col-sum err {col_dev.abs().max().item()}"
    ref_row = (comb_ref.reshape(-1, n, n).sum(dim=-1) - 1).abs().max().item()
    assert (m.sum(dim=-1) - 1).abs().max().item() <= ref_row + 5e-5


@pytest.mark.parametrize("x_shape", SHAPES, ids=lambda s: "X" + "x".join(map(str, s)))
def test_mhc_pre(device, x_shape):
    dtype = ttnn.float32
    x, w, b, scale = make_inputs(x_shape)
    y, post, comb = mhc_pre(
        to_dev(x, device, dtype), to_dev(w, device, ttnn.float32), to_dev(b, device, ttnn.float32), scale=scale
    )
    _check(y, post, comb, x, w, b, scale, dtype)


@pytest.mark.parametrize("iters", [1, 5])
def test_mhc_pre_sinkhorn_iters_and_config(device, iters):
    """Non-default iteration count and an explicit compute config."""
    dtype = ttnn.float32
    x_shape = (1, 96, 4 * 512)
    x, w, b, _ = make_inputs(x_shape, seed=7)
    scale = (0.7, 1.3, 2.0)
    cfg = default_compute_kernel_config()
    y, post, comb = mhc_pre(
        to_dev(x, device, dtype),
        to_dev(w, device, ttnn.float32),
        to_dev(b, device, ttnn.float32),
        scale=scale,
        sinkhorn_iters=iters,
        compute_kernel_config=cfg,
    )
    _check(y, post, comb, x, w, b, scale, dtype, iters=iters)


def test_mhc_pre_large_logits_overflow_safe(device):
    """Logits of magnitude ~80+ must not produce Inf/NaN; comb stays column-stochastic."""
    n = N_HC
    x, w, b, _ = make_inputs((1, 64, 4 * 256), seed=11)
    scale = (1.0, 1.0, 40.0)
    y, post, comb = mhc_pre(
        to_dev(x, device, ttnn.float32), to_dev(w, device, ttnn.float32), to_dev(b, device, ttnn.float32), scale=scale
    )
    for t in (y, post, comb):
        assert torch.isfinite(ttnn.to_torch(t)).all()
    m = ttnn.to_torch(comb).to(torch.float64).reshape(-1, n, n)
    assert (m >= 0).all()
    assert (m.sum(dim=-2) - 1).abs().max().item() < 5e-5


def test_mhc_pre_deterministic(device):
    """Two calls on identical inputs are bitwise identical (fixed cross-core reduction order)."""
    x, w, b, scale = make_inputs((1, 1, 640, 4 * 1792), seed=3)
    tx = to_dev(x, device, ttnn.float32)
    tw = to_dev(w, device, ttnn.float32)
    tb = to_dev(b, device, ttnn.float32)
    out1 = [ttnn.to_torch(t) for t in mhc_pre(tx, tw, tb, scale=scale)]
    out2 = [ttnn.to_torch(t) for t in mhc_pre(tx, tw, tb, scale=scale)]
    for a, c in zip(out1, out2):
        assert torch.equal(
            a.view(torch.int32) if a.dtype == torch.float32 else a,
            c.view(torch.int32) if c.dtype == torch.float32 else c,
        )


def test_mhc_pre_refuses_16bit_dest(device, expect_error):
    x, w, b, scale = make_inputs((32, 4 * 32))
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
    )
    with expect_error(UnsupportedAxisValue, "fp32_dest_acc_en"):
        mhc_pre(
            to_dev(x, device, ttnn.float32),
            to_dev(w, device, ttnn.float32),
            to_dev(b, device, ttnn.float32),
            scale=scale,
            compute_kernel_config=cfg,
        )


def test_default_compute_kernel_config():
    cfg = default_compute_kernel_config()
    assert cfg.fp32_dest_acc_en is True
    assert cfg.math_approx_mode is False
    assert cfg.math_fidelity == ttnn.MathFidelity.HiFi4
