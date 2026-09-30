# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for mhc_pre golden tests.

Provides:
- pytorch_mhc_pre: reference (y, post, comb) at the mHC paper's precision.
- sinkhorn_knopp: the DeepSeek-V4 Sinkhorn projection used by the reference.
- round_to_tf32: RNE rounding of a weight to tfloat32 (the paper's phi dtype).
- create_ttnn_input_tensor: single chokepoint for torch -> ttnn conversion.
- make_inputs: seeded X / W / bias / scale for one case.
- TOLERANCES: keyed by (output, stream dtype).
- check_doubly_stochastic: structural gate on the comb output.
- run_mhc_pre: canonical entry point used by test_golden.py.

Shape/dtype/layout + PCC/RMS checking lives in `eval/metrics.py`.
run_mhc_pre dispatches through axes.observed (aliased as mhc_pre).
"""

from __future__ import annotations

import torch
import ttnn

from eval.metrics import _CONTRACT_METRICS, CheckOutputError, Tolerance, check_output  # noqa: F401

from ttnn.bringup.mhc_pre_ttnn.tests.golden.axes import observed as mhc_pre  # type: ignore


# ---------------------------------------------------------------------------
# Reference — precision contract
# ---------------------------------------------------------------------------
#
# The reference follows the per-tensor dtypes of DeepSeek's own fused mHC
# kernel (Xie et al., "mHC: Manifold-Constrained Hyper-Connections",
# arXiv 2512.24880, §4.3.1, Eqs. 10-19):
#
#   x (the n streams, flattened)   bfloat16 [1, nC]       (Eq. 11)
#   phi (projection weight W)      tfloat32 [nC, n^2+2n]  (Eq. 10)
#   alpha, b                       float32                (Eqs. 12-13)
#   H~ (raw mixes), r (RMS)        float32                (Eqs. 14-16)
#   H_pre, H_post, H_res (comb)    float32                (Eqs. 17-19)
#
# bfloat16 is the production stream dtype (DeepSeek-V4, GLM-5.3); float32
# streams are the golden suite's accuracy ceiling. W is rounded to tf32 with
# round-to-nearest-even ONCE, when it is loaded (make_inputs here): the op
# may then read it through the FPU's tf32 source registers losslessly, and
# the reference sees exactly those weight values. Everything after the
# inputs is computed in float32, like the paper's kernel.


def hc_mult_from_weight(w_shape):
    """n from W's last dim n*(n+2) (24 -> 4)."""
    mix = w_shape[-1]
    n = int(round((mix + 1) ** 0.5 - 1))
    if n * (n + 2) != mix:
        raise ValueError(f"W last dim {mix} is not n*(n+2) for any integer n")
    return n


def round_to_tf32(t):
    """Round float32 values to tfloat32 (10-bit mantissa), round-to-nearest-even.

    Returned as float32 (tf32 is a subset). bfloat16 values are already
    tf32-representable and pass through unchanged.
    """
    b = t.to(torch.float32).contiguous().view(torch.int32)
    drop = 13  # float32 keeps 23 mantissa bits, tf32 keeps 10
    b = b + ((1 << (drop - 1)) - 1) + ((b >> drop) & 1)
    return (b & ~((1 << drop) - 1)).view(torch.float32)


def sinkhorn_knopp(logits, iters, eps):
    """Project a batch of n x n matrices onto the doubly-stochastic manifold.

    Exactly DeepSeek-V4 inference/kernel.py:hc_split_sinkhorn_kernel:
    row softmax (+eps), one column normalisation, then (iters - 1)
    alternations of row- then column-normalisation. Ends on a column
    normalisation: columns sum to 1, rows to 1 within ~eps.
    """
    m = torch.softmax(logits, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


def pytorch_mhc_pre(input_tensor, proj_weight, proj_bias, *, scale, sinkhorn_iters=20, eps=1e-6, norm_eps=1e-6):
    """Reference (y, post, comb) at the paper's precision (returned float32).

    ORACLE RULE — pure torch, MUST NOT call any ttnn op. No torch built-in
    computes mHC, so this decomposes DeepSeek-V4 inference/model.py
    Block.hc_pre + kernel.py hc_split_sinkhorn_kernel into torch primitives.

    input_tensor (..., T, n*C) as stored on device (bf16 or fp32 streams);
    proj_weight (n*C, n*(n+2)), already tf32-representable (make_inputs);
    proj_bias (1, n*(n+2)); scale = (a_pre, a_post, a_res). All arithmetic
    is float32 (the precision contract above).
    Returns y (..., T, C), post (..., T, n), comb (..., T, n*n) with
    comb[..., i*n + j] = comb[i][j].
    """
    n = hc_mult_from_weight(proj_weight.shape)
    lead, nc = tuple(input_tensor.shape[:-1]), input_tensor.shape[-1]
    C = nc // n
    x = input_tensor.to(torch.float32).reshape(-1, nc)
    w = proj_weight.to(torch.float32)
    b = proj_bias.to(torch.float32).reshape(-1)
    a_pre, a_post, a_res = (float(s) for s in scale)

    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = (x @ w) * rsqrt

    pre = torch.sigmoid(mixes[:, 0:n] * a_pre + b[0:n]) + eps
    post = 2.0 * torch.sigmoid(mixes[:, n : 2 * n] * a_post + b[n : 2 * n])
    logits = (mixes[:, 2 * n :] * a_res + b[2 * n :]).reshape(-1, n, n)
    comb = sinkhorn_knopp(logits, sinkhorn_iters, eps).reshape(-1, n * n)

    y = (pre.unsqueeze(-1) * x.reshape(-1, n, C)).sum(dim=1)
    return (y.reshape(*lead, C), post.reshape(*lead, n), comb.reshape(*lead, n * n))


# EQUIVALENCE GATE (run once at authoring, 2026-09-29): pytorch_mhc_pre vs
# models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py
# MHCWrap.hc_pre (the DeepSeek-V4 port) in float64 on seeded inputs,
# shapes (2, 3, 4*32) / (1, 5, 4*64) / (1, 1, 4*128) with random base and
# scale (3 seeds each) — max relative diff 1.3e-7 (y) / 1.4e-7 (post) /
# 1.9e-7 (comb), i.e. the fp32 rounding of this oracle; bound 1e-6.


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


def create_ttnn_input_tensor(tensor, device, *, dtype, layout, memory_config=None):
    """Single chokepoint for `torch.Tensor -> ttnn.Tensor`. Don't bypass
    this with `ttnn.from_torch(...)` in test code."""
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def make_inputs(
    x_shape, w_shape, *, dtype, weight_dtype, seed=0, x_scale=1.0, logit_scale=1.0, identical_streams=False
):
    """Seeded X, W, bias, scale — already at their device precision.

    X ~ N(0, x_scale^2), rounded to the stream dtype. W ~ N(0, 1/(n*C)) so
    each raw mix is ~N(0, 1) after the RMSNorm (whatever x_scale is); a
    float32 W is rounded to tf32 (RNE) — the paper's phi dtype — and a
    bfloat16 W is rounded to bf16. identical_streams makes all n streams
    equal — what mhc_expand feeds the first layer. bias ~ N(0, 1) and
    scale = (1, 1, logit_scale): the Sinkhorn logits are O(logit_scale),
    so the comb is far from uniform and exercises the iteration.
    """
    g = torch.Generator().manual_seed(seed)
    K, mix = w_shape
    x = torch.randn(x_shape, generator=g) * x_scale
    if identical_streams:
        n = hc_mult_from_weight(w_shape)
        C = x_shape[-1] // n
        x = x[..., :C].repeat(*([1] * (len(x_shape) - 1)), n)
    x = x.to(_TORCH_DTYPE[dtype])
    w = torch.randn(w_shape, generator=g) / K**0.5
    w = round_to_tf32(w) if weight_dtype == ttnn.float32 else w.to(_TORCH_DTYPE[weight_dtype])
    bias = torch.randn((1, mix), generator=g)
    scale = (1.0, 1.0, float(logit_scale))
    return x, w, bias, scale


def make_compute_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# Tolerances — keyed by (output, stream dtype); DEST is always fp32
# ---------------------------------------------------------------------------

# rms is relative to the reference stddev (eval/metrics.py). The reference
# sees the same rounded inputs, so each budget covers only the op's own
# arithmetic:
#   bf16 streams: X and the tf32 W are exact in the FPU source registers;
#     the n*C-deep projection accumulates in fp32 DEST, so post / comb carry
#     ~fp32 accumulation noise only. y is rounded to bf16 (rel ~1e-3).
#   fp32 streams: the FPU reads X at tf32 (by truncation), so the projection
#     — and so post / comb — sit at tf32-class noise (emulated rel ~8e-4).
#     This bias is harmless (mixes only feed sigmoid / Sinkhorn, and y feeds
#     a sublayer that re-normalises), so only the noise level is gated.
#     y itself can be exact (SFPU mixing) or tf32-class (FPU mixing).
# NOTE: set from host emulation, not yet calibrated on device —
# recalibrate on the first HW run.
TOLERANCES = {
    ("y", ttnn.float32): (0.99999, 2e-3),
    ("y", ttnn.bfloat16): (0.99995, 4e-3),
    ("coeff", ttnn.float32): (0.99999, 2e-3),
    ("coeff", ttnn.bfloat16): (0.999999, 5e-4),
}

# Doubly-stochastic gate on comb (absolute), every regime: the Sinkhorn is
# cheap ([T, 16]) and always runs at full fp32. Mean signed column-sum error
# over all tokens and columns: a bias composes over depth (122 wraps in
# DeepSeek-V4: drift ~ 122 * bias), fp32 Sinkhorn gives ~1e-6 (the eps), a
# truncating tf32 normalisation ~-5e-4. Max: worst single column.
# Rows may exceed the REFERENCE's own row error by the slack (the reference
# is not exactly row-stochastic after finite iterations).
COL_SUM_MEAN_BIAS = 1e-5
COL_SUM_MAX = 5e-5
ROW_SUM_SLACK = 5e-5


def check_doubly_stochastic(comb_out, comb_ref, *, n):
    """comb must be doubly stochastic like the reference. severity=bug."""
    got = ttnn.to_torch(comb_out).to(torch.float64).reshape(-1, n, n)
    ref = comb_ref.to(torch.float64).reshape(-1, n, n)
    col_dev = got.sum(dim=-2) - 1
    mean_bias = col_dev.mean().item()
    col_max = col_dev.abs().max().item()
    row_err = (got.sum(dim=-1) - 1).abs().max().item()
    ref_row_err = (ref.sum(dim=-1) - 1).abs().max().item()
    neg = bool((got < 0).any())
    if neg or abs(mean_bias) > COL_SUM_MEAN_BIAS or col_max > COL_SUM_MAX or row_err > ref_row_err + ROW_SUM_SLACK:
        raise CheckOutputError(
            severity="bug",
            metrics=_CONTRACT_METRICS,
            tolerance=Tolerance(pcc=None, rms=None),
            extra=(
                f"comb_not_doubly_stochastic: negative={neg} "
                f"mean(colsum-1)={mean_bias:+.3g} (limit {COL_SUM_MEAN_BIAS}) "
                f"max|colsum-1|={col_max:.3g} (limit {COL_SUM_MAX}) "
                f"max|rowsum-1|={row_err:.3g} (reference {ref_row_err:.3g})"
            ),
        )


# ---------------------------------------------------------------------------
# run_mhc_pre — canonical per-axes dispatch
# ---------------------------------------------------------------------------


def run_mhc_pre(inputs, *, dtype, layout, weight_dtype, device, extras=None, **_):
    """Build X/W/bias, dispatch mhc_pre, check all three outputs. Raises
    CheckOutputError on a numerical/contract miss.

    `inputs` is `(X_shape, W_shape)`. Outputs: y `(..., T, C)` in `dtype`,
    post `(..., T, n)` and comb `(..., T, n*n)` in float32, all TILE.
    fp32_dest_acc_en is always True (the only TARGET value), passed
    explicitly so the op's config handling is exercised.

    `extras` (loose / regression cases) may carry `seed`, `x_scale`,
    `logit_scale`, `identical_streams`; perf goals in it are recorded by the
    harness, not asserted here. Returns (y, post, comb) as torch float64 for
    callers that compose outputs (the depth regression tests).
    """
    extras = extras or {}
    x_shape, w_shape = tuple(inputs[0]), tuple(inputs[1])
    n = hc_mult_from_weight(w_shape)
    C = x_shape[-1] // n
    lead = x_shape[:-1]

    x, w, bias, scale = make_inputs(
        x_shape,
        w_shape,
        dtype=dtype,
        weight_dtype=weight_dtype,
        seed=extras.get("seed", 0),
        x_scale=extras.get("x_scale", 1.0),
        logit_scale=extras.get("logit_scale", 1.0),
        identical_streams=extras.get("identical_streams", False),
    )
    y_ref, post_ref, comb_ref = pytorch_mhc_pre(x, w, bias, scale=scale)

    ttnn_x = create_ttnn_input_tensor(x, device, dtype=dtype, layout=layout)
    ttnn_w = create_ttnn_input_tensor(w, device, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
    ttnn_b = create_ttnn_input_tensor(bias, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    y, post, comb = mhc_pre(ttnn_x, ttnn_w, ttnn_b, scale=scale, compute_kernel_config=make_compute_config())

    check_output(
        y,
        y_ref.to(_TORCH_DTYPE[dtype]),
        shape=list(lead) + [C],
        dtype=dtype,
        expected_layout=layout,
        tolerance=TOLERANCES[("y", dtype)],
    )
    check_output(
        post,
        post_ref,
        shape=list(lead) + [n],
        dtype=ttnn.float32,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=TOLERANCES[("coeff", dtype)],
    )
    check_output(
        comb,
        comb_ref,
        shape=list(lead) + [n * n],
        dtype=ttnn.float32,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=TOLERANCES[("coeff", dtype)],
    )
    check_doubly_stochastic(comb, comb_ref, n=n)
    return tuple(ttnn.to_torch(t).to(torch.float64) for t in (y, post, comb))
