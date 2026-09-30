# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for mhc_post golden tests.

Provides:
- pytorch_mhc_post: reference X' at the mHC paper's precision.
- create_ttnn_input_tensor: single chokepoint for torch -> ttnn conversion.
- make_inputs: seeded F / X / post / comb for one case (comb doubly
  stochastic, post in (0, 2) — the distributions mhc_pre produces).
- TOLERANCES: keyed by the stream (output) dtype.
- check_signed_bias: gate on systematic shrink / growth of X'.
- run_mhc_post: canonical entry point used by test_golden.py.
- run_mhc_post_chain: X' fed back as X for `depth` calls (depth tests).

Shape/dtype/layout + PCC/RMS checking lives in `eval/metrics.py`.
run_mhc_post dispatches through axes.observed (aliased as mhc_post).
"""

from __future__ import annotations

import torch
import ttnn

from eval.metrics import _CONTRACT_METRICS, CheckOutputError, Tolerance, check_output  # noqa: F401

from ttnn.bringup.mhc_post_ttnn.tests.golden.axes import observed as mhc_post  # type: ignore


# ---------------------------------------------------------------------------
# Reference — precision contract
# ---------------------------------------------------------------------------
#
# Per-tensor dtypes follow DeepSeek's fused mHC kernel (Xie et al., "mHC:
# Manifold-Constrained Hyper-Connections", arXiv 2512.24880, §4.3.1): the
# residual streams are bfloat16 in production (Eq. 11; DeepSeek-V4 and
# GLM-5.3 both run bf16 streams), H_post and H_res are float32 (Eqs. 18-19).
# float32 streams are the golden suite's accuracy ceiling. The reference
# computes the mix in float32 from the inputs exactly as stored on device and
# casts X' to the stream dtype once, at the end. (The Hugging Face ports cast
# post / comb to the stream dtype before mixing; the paper's kernel keeps
# them float32, which this reference follows.)
#
# X' is the residual highway: it is fed back as the next wrap's X, 2 wraps
# per layer (122 in DeepSeek-V4). Any systematic shrink compounds with
# depth, so on top of PCC / RMS the output is gated on its SIGNED relative
# bias, and test_regression.py composes the op over 122 calls.


def pytorch_mhc_post(input_tensor, residual, post, comb):
    """Reference X' computed in float32 (returned float32).

    ORACLE RULE — pure torch, MUST NOT call any ttnn op. No torch built-in
    computes mHC, so this is DeepSeek-V4 inference/model.py Block.hc_post in
    torch primitives. Per output stream j:

        X'_j = post_j * F + sum_i comb[i][j] * X_i

    input_tensor F (..., T, C); residual X (..., T, n*C); post (..., T, n);
    comb (..., T, n*n) with comb[..., i*n + j] = comb[i][j].
    """
    n = post.shape[-1]
    lead, C = tuple(input_tensor.shape[:-1]), input_tensor.shape[-1]
    f = input_tensor.to(torch.float32).reshape(-1, 1, C)
    x = residual.to(torch.float32).reshape(-1, n, C)
    p = post.to(torch.float32).reshape(-1, n, 1)
    m = comb.to(torch.float32).reshape(-1, n, n)
    out = p * f + torch.einsum("tij,tic->tjc", m, x)
    return out.reshape(*lead, n * C)


# EQUIVALENCE GATE (run once at authoring, 2026-09-29): pytorch_mhc_post vs
# models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py
# MHCWrap.hc_post in float64 on seeded inputs, shapes (2, 3, 4*32) /
# (1, 5, 4*64) / (1, 1, 4*128), 3 seeds each — max relative diff 9.6e-8
# (the fp32 rounding of this oracle); bound 1e-6.


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


def _sinkhorn(logits, iters=20, eps=1e-6):
    """DeepSeek-V4 Sinkhorn (same as mhc_pre's) — makes realistic combs."""
    m = torch.softmax(logits, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


def make_coefficients(post_shape, comb_shape, *, generator, comb_mode="sinkhorn"):
    """post = 2*sigmoid(N(0, 1)) and a comb, both float32 (mhc_pre's output dtype).

    comb_mode: "sinkhorn" (Sinkhorn of N(0, 1) logits — mhc_pre's output
    distribution), "identity" (pure residual pass-through per stream),
    "cyclic" (comb[i][(i+1) % n] = 1, so X'_j carries X_{j-1}: pins the
    comb orientation) or "uniform" (every stream becomes the stream mean).
    """
    n = post_shape[-1]
    post = 2.0 * torch.sigmoid(torch.randn(post_shape, generator=generator))
    lead = comb_shape[:-1]
    if comb_mode == "sinkhorn":
        comb = _sinkhorn(torch.randn(*lead, n, n, generator=generator))
    elif comb_mode == "identity":
        comb = torch.eye(n).expand(*lead, n, n)
    elif comb_mode == "cyclic":
        comb = torch.roll(torch.eye(n), shifts=1, dims=1).expand(*lead, n, n)
    elif comb_mode == "uniform":
        comb = torch.full((*lead, n, n), 1.0 / n)
    else:
        raise ValueError(comb_mode)
    return post.float(), comb.reshape(comb_shape).float().contiguous()


def make_inputs(
    f_shape,
    x_shape,
    post_shape,
    comb_shape,
    *,
    dtype,
    sublayer_dtype,
    seed=0,
    f_scale=1.0,
    x_scale=1.0,
    comb_mode="sinkhorn",
):
    """Seeded F, X, post, comb — rounded to their device dtypes here so the
    reference sees exactly what the device does. F ~ N(0, f_scale^2),
    X ~ N(0, x_scale^2); coefficients per make_coefficients."""
    g = torch.Generator().manual_seed(seed)
    f = (torch.randn(f_shape, generator=g) * f_scale).to(_TORCH_DTYPE[sublayer_dtype])
    x = (torch.randn(x_shape, generator=g) * x_scale).to(_TORCH_DTYPE[dtype])
    post, comb = make_coefficients(post_shape, comb_shape, generator=g, comb_mode=comb_mode)
    return f, x, post, comb


def make_compute_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# Tolerances and the bias gate — keyed by the stream (output) dtype
# ---------------------------------------------------------------------------

# The reference sees the same rounded inputs, so the budget covers only the
# op's arithmetic and the final rounding of X' to the stream dtype (the
# sublayer dtype changes the inputs, not the error budget). rms is relative
# to the reference stddev (eval/metrics.py).
#   float32 X': exact fp32 mixing (SFPU, or FPU with lossless operands)
#     reaches rel ~2e-7; an FPU reading fp32 operands at tf32 does not.
#   bfloat16 X': one bf16 rounding of the result (rel ~1e-3).
# NOTE: set from host emulation, not yet calibrated on device —
# recalibrate on the first HW run.
TOLERANCES = {
    ttnn.float32: (0.9999999, 2e-6),
    ttnn.bfloat16: (0.99995, 4e-3),
}

# Signed relative bias of X': mean((X' - ref) * sign(ref)) / mean|ref|,
# against the UNROUNDED fp32 reference. Emulated per call: exact fp32 mixing
# 1e-10; bf16 output rounding ~+-6e-6 (sampling noise of an unbiased
# rounding); FPU operands truncated to tf32 -7e-4 (fp32 streams) / -3.4e-4
# (bf16 streams), which over 122 wraps shrinks the residual by 2.5-5%.
# The gate is |bias| <= floor + BIAS_SIGMAS * standard error: rounding noise
# averages out as 1/sqrt(N), so the allowance is measured from the data and
# small shapes are not failed by noise, while a real bias (fixed in N) is.
SIGNED_BIAS_FLOOR = {
    ttnn.float32: 1e-6,
    ttnn.bfloat16: 1e-5,
}
BIAS_SIGMAS = 6.0


def signed_bias(actual, expected):
    """(bias, standard error) of the signed relative error; bias < 0 when
    the result is systematically shrunk."""
    a, e = actual.to(torch.float64).flatten(), expected.to(torch.float64).flatten()
    d = (a - e) * e.sign()
    scale = e.abs().mean().clamp_min(1e-30)
    return (d.mean() / scale).item(), (d.std() / d.numel() ** 0.5 / scale).item()


def check_signed_bias(out, expected_fp32, *, dtype):
    """X' must not be systematically shrunk or grown. severity=bug."""
    bias, se = signed_bias(ttnn.to_torch(out), expected_fp32)
    limit = SIGNED_BIAS_FLOOR[dtype] + BIAS_SIGMAS * se
    if abs(bias) > limit:
        raise CheckOutputError(
            severity="bug",
            metrics=_CONTRACT_METRICS,
            tolerance=Tolerance(pcc=None, rms=None),
            extra=(
                f"residual_biased: signed relative bias {bias:+.3g} > {limit:.3g} "
                f"(floor {SIGNED_BIAS_FLOOR[dtype]} + {BIAS_SIGMAS:g} x se {se:.2g}); "
                f"composes over depth"
            ),
        )


# ---------------------------------------------------------------------------
# run_mhc_post — canonical per-axes dispatch
# ---------------------------------------------------------------------------


def run_mhc_post(inputs, *, dtype, sublayer_dtype, layout, device, extras=None, **_):
    """Build F/X/post/comb, dispatch mhc_post, check. Raises
    CheckOutputError on a numerical/contract miss.

    `inputs` is `(F_shape, X_shape, post_shape, comb_shape)`. Output X' has
    X's shape, `dtype`, TILE. fp32_dest_acc_en is always True (the only
    TARGET value), passed explicitly so the op's config handling is exercised.

    `extras` (loose / regression cases) may carry `seed`, `f_scale`,
    `x_scale`, `comb_mode`; perf goals in it are recorded by the harness,
    not asserted here.
    """
    extras = extras or {}
    f_shape, x_shape, post_shape, comb_shape = (tuple(s) for s in inputs)

    f, x, post, comb = make_inputs(
        f_shape,
        x_shape,
        post_shape,
        comb_shape,
        dtype=dtype,
        sublayer_dtype=sublayer_dtype,
        seed=extras.get("seed", 0),
        f_scale=extras.get("f_scale", 1.0),
        x_scale=extras.get("x_scale", 1.0),
        comb_mode=extras.get("comb_mode", "sinkhorn"),
    )
    expected = pytorch_mhc_post(f, x, post, comb)

    out = mhc_post(
        create_ttnn_input_tensor(f, device, dtype=sublayer_dtype, layout=layout),
        create_ttnn_input_tensor(x, device, dtype=dtype, layout=layout),
        create_ttnn_input_tensor(post, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
        create_ttnn_input_tensor(comb, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
        compute_kernel_config=make_compute_config(),
    )
    check_output(
        out,
        expected.to(_TORCH_DTYPE[dtype]),
        shape=list(x_shape),
        dtype=dtype,
        expected_layout=layout,
        tolerance=TOLERANCES[dtype],
    )
    check_signed_bias(out, expected, dtype=dtype)


def run_mhc_post_chain(x_shape, *, dtype, depth, device, seed=0, f_scale=0.1):
    """Compose mhc_post `depth` times, feeding X' back as X, next to a float64
    reference chain from the same start. Each wrap draws a fresh sublayer
    output F ~ N(0, f_scale^2) (sublayer dtype = stream dtype) and fresh
    post / comb (Sinkhorn of N(0, 1) logits).

    Returns (norm_drift, rel_err) after `depth` wraps:
      norm_drift = ||X_dev|| / ||X_ref|| - 1   (< 0: the highway shrank)
      rel_err    = ||X_dev - X_ref|| / ||X_ref||
    """
    g = torch.Generator().manual_seed(seed)
    n = 4
    lead, nc = tuple(x_shape[:-1]), x_shape[-1]
    C = nc // n
    x0 = torch.randn(x_shape, generator=g).to(_TORCH_DTYPE[dtype])
    x_ref = x0.to(torch.float64)
    x_dev = create_ttnn_input_tensor(x0, device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    for _ in range(depth):
        f = (torch.randn(lead + (C,), generator=g) * f_scale).to(_TORCH_DTYPE[dtype])
        post, comb = make_coefficients(lead + (n,), lead + (n * n,), generator=g)
        m = comb.double().reshape(-1, n, n)
        x_ref = (
            post.double().reshape(-1, n, 1) * f.double().reshape(-1, 1, C)
            + torch.einsum("tij,tic->tjc", m, x_ref.reshape(-1, n, C))
        ).reshape(x_shape)
        x_dev = mhc_post(
            create_ttnn_input_tensor(f, device, dtype=dtype, layout=ttnn.TILE_LAYOUT),
            x_dev,
            create_ttnn_input_tensor(post, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
            create_ttnn_input_tensor(comb, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
            compute_kernel_config=make_compute_config(),
        )
    got = ttnn.to_torch(x_dev).to(torch.float64)
    return ((got.norm() / x_ref.norm() - 1).item(), ((got - x_ref).norm() / x_ref.norm()).item())
