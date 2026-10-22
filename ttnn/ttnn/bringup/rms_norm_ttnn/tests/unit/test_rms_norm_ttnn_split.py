# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CHANGELOG 6 -- the split program: `stats_out`, `partial_stats` / `stats_width`, `input_window`, `dtype`.

DO NOT DELETE.  What this pins (single device; the multi-chip use is an all_gather of `stats_out` between the two
calls, emulated here by a ttnn.concat of the partials of two column halves):

  1. stats_out: every element of the [.., R, 32 B] float32 output is a partial sum of x^2; a row's elements sum to
     the row's sum of squares (vs float64, FPU-grade: x^2 read at TF32).
  2. partial_stats: y = x * rsqrt(sum of the row's partials / stats_width + eps) * w over a split row equals the
     whole-row RMSNorm (vs float64); with the wrong stats_width the same check fails (the test is sensitive).
  3. input_window: only the window's columns are read and normalized (local statistic), any dtype out.
  4. the refusals of the split options.

Limits: rel L2 vs float64 <= 1.25 x one rounding of the reference to the output dtype, never below 1.5e-3 (fp32 out:
the FPU statistic and x * rs at TF32 measured 2.7e-4 .. 9e-4); the median row-norm ratio within 1e-3 of 1 (measured
-1.6e-4 .. -7e-4, ds41_norm_split probe_002).
"""

from __future__ import annotations

import pytest
import torch

import ttnn

D = ttnn.DRAM_MEMORY_CONFIG
EPS = 1e-6


def _ck(device, fp32=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=False,
    )


def _up(device, t, dtype):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)


def _ref(x, w):
    x = x.double()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y if w is None else y * w.double()


def _check(got, want, out_dtype):
    got = got.double()
    rel = ((got - want).norm() / want.norm()).item()
    floor = (
        (want.float().to(torch.bfloat16 if out_dtype == ttnn.bfloat16 else torch.float32).double() - want).norm()
        / want.norm()
    ).item()
    ratio = (got.norm(dim=-1) / want.norm(dim=-1)).median().item() - 1.0
    limit = max(1.25 * floor, 1.5e-3)
    return rel, ratio, limit


def _inputs(rows, width, in_dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, rows, width, generator=g) * (torch.rand(1, 1, rows, 1, generator=g) * 3 + 0.1)
    if in_dtype == ttnn.bfloat16:
        x = x.bfloat16().float()
    w = (torch.rand(width, generator=g) + 0.5).bfloat16().float()
    return x, w


@pytest.mark.parametrize("in_dtype", [ttnn.float32, ttnn.bfloat16], ids=["x_fp32", "x_bf16"])
@pytest.mark.parametrize("rows, width", [(64, 256), (256, 1280), (96, 4096)])
def test_stats_out(device, in_dtype, rows, width):
    x, _ = _inputs(rows, width, in_dtype)
    s = ttnn.bringup.rms_norm(
        _up(device, x, in_dtype), stats_out=True, compute_kernel_config=_ck(device), memory_config=D
    )
    assert s.dtype == ttnn.float32 and s.layout == ttnn.TILE_LAYOUT
    assert s.shape[-2] == rows and s.shape[-1] % 32 == 0, s.shape
    got = ttnn.to_torch(s).double().sum(-1)
    want = x.double().pow(2).sum(-1)
    rel = ((got - want).norm() / want.norm()).item()
    assert rel <= 2e-3, rel


@pytest.mark.parametrize("out_dtype", [ttnn.float32, ttnn.bfloat16], ids=["out_fp32", "out_bf16"])
@pytest.mark.parametrize("in_dtype", [ttnn.float32, ttnn.bfloat16], ids=["x_fp32", "x_bf16"])
@pytest.mark.parametrize("weight", [True, False], ids=["w", "no_w"])
@pytest.mark.parametrize("rows, half", [(64, 128), (256, 2560), (1024, 640)])
def test_partial_stats_split_row(device, out_dtype, in_dtype, weight, rows, half):
    """A row split into two column halves: stats_out per half, the partials concatenated (the all_gather of a
    2-chip mesh), each half normalized by the whole row's statistic."""
    x, w = _inputs(rows, 2 * half, in_dtype)
    ck = _ck(device)
    halves = [x[..., :half], x[..., half:]]
    xd = [_up(device, h, in_dtype) for h in halves]
    parts = [ttnn.bringup.rms_norm(t, stats_out=True, compute_kernel_config=ck, memory_config=D) for t in xd]
    gathered = ttnn.concat(parts, dim=-1, memory_config=D)
    want = _ref(x, w if weight else None)
    for i, t in enumerate(xd):
        kw = dict(
            partial_stats=gathered,
            stats_width=2 * half,
            epsilon=EPS,
            dtype=out_dtype,
            compute_kernel_config=ck,
            memory_config=D,
        )
        if weight:
            kw["weight"] = _up(device, w[i * half : (i + 1) * half].reshape(1, 1, 1, half), ttnn.float32)
        y = ttnn.bringup.rms_norm(t, **kw)
        assert y.dtype == out_dtype and list(y.shape) == [1, 1, rows, half], (y.dtype, y.shape)
        rel, ratio, limit = _check(ttnn.to_torch(y), want[..., i * half : (i + 1) * half], out_dtype)
        assert rel <= limit and abs(ratio) <= 1e-3, (rel, limit, ratio)
        # sensitivity: the half-row width as stats_width (a wrong statistic) must fail the same check
        kw["stats_width"] = half
        bad = ttnn.bringup.rms_norm(t, **kw)
        rel_b, _, _ = _check(ttnn.to_torch(bad), want[..., i * half : (i + 1) * half], out_dtype)
        assert rel_b > 10 * limit, (rel_b, limit)


@pytest.mark.parametrize("out_dtype", [ttnn.float32, ttnn.bfloat16], ids=["out_fp32", "out_bf16"])
@pytest.mark.parametrize("in_dtype", [ttnn.float32, ttnn.bfloat16], ids=["x_fp32", "x_bf16"])
@pytest.mark.parametrize("window", [(0, 1280), (1280, 1792), (0, 1792), (256, 288)])
def test_input_window(device, out_dtype, in_dtype, window):
    """The fused-projection case: q_norm / kv_norm read their columns of one [R, 1792] tensor in place."""
    rows = 128
    x, _ = _inputs(rows, 1792, in_dtype, seed=1)
    c0, c1 = window
    w = (torch.rand(c1 - c0) + 0.5).bfloat16().float()
    y = ttnn.bringup.rms_norm(
        _up(device, x, in_dtype),
        weight=_up(device, w.reshape(1, 1, 1, -1), ttnn.float32),
        input_window=window,
        dtype=out_dtype,
        epsilon=EPS,
        compute_kernel_config=_ck(device),
        memory_config=D,
    )
    assert y.dtype == out_dtype and list(y.shape) == [1, 1, rows, c1 - c0], (y.dtype, y.shape)
    rel, ratio, limit = _check(ttnn.to_torch(y), _ref(x[..., c0:c1], w), out_dtype)
    assert rel <= limit and abs(ratio) <= 1e-3, (rel, limit, ratio)


def test_dtype_only_takes_the_split_program(device):
    """dtype alone (no window, no stats) normalizes the whole row and writes the requested dtype."""
    x, w = _inputs(64, 512, ttnn.float32, seed=2)
    y = ttnn.bringup.rms_norm(
        _up(device, x, ttnn.float32),
        weight=_up(device, w.reshape(1, 1, 1, -1), ttnn.float32),
        dtype=ttnn.bfloat16,
        epsilon=EPS,
        compute_kernel_config=_ck(device),
        memory_config=D,
    )
    assert y.dtype == ttnn.bfloat16
    rel, ratio, limit = _check(ttnn.to_torch(y), _ref(x, w), ttnn.bfloat16)
    assert rel <= limit and abs(ratio) <= 1e-3, (rel, limit, ratio)


def test_split_options_refusals(device, expect_error):
    x, w = _inputs(64, 256, ttnn.float32, seed=3)
    xd = _up(device, x, ttnn.float32)
    ck = _ck(device)
    s = ttnn.bringup.rms_norm(xd, stats_out=True, compute_kernel_config=ck, memory_config=D)
    with expect_error(Exception, "must be tile-aligned inside"):  # window not tile-aligned
        ttnn.bringup.rms_norm(xd, input_window=(16, 128), compute_kernel_config=ck)
    with expect_error(Exception, "outside"):  # window past the row
        ttnn.bringup.rms_norm(xd, input_window=(0, 288), compute_kernel_config=ck)
    with expect_error(Exception, "exclude each other"):  # stats_out and partial_stats together
        ttnn.bringup.rms_norm(xd, stats_out=True, partial_stats=s, compute_kernel_config=ck)
    with expect_error(Exception, "stats_out takes no weight"):
        ttnn.bringup.rms_norm(
            xd, stats_out=True, weight=_up(device, w.reshape(1, 1, 1, -1), ttnn.float32), compute_kernel_config=ck
        )
    with expect_error(Exception, "needs fp32_dest_acc_en"):  # the fp32 statistic needs fp32 DEST
        ttnn.bringup.rms_norm(xd, stats_out=True, compute_kernel_config=_ck(device, fp32=False))
    with expect_error(ValueError, "take no bias, residual"):  # no bias / residual with the split options
        ttnn.bringup.rms_norm(xd, residual_input_tensor=xd, dtype=ttnn.bfloat16, compute_kernel_config=ck)
