# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for mhc_pre (verifier-authored).

Measures, per output (y, post, comb), against the golden fp32 reference
(eval/golden_tests/mhc_pre/helpers.py: pytorch_mhc_pre, same rounded inputs):
PCC, max / mean abs error, relative RMS error (vs reference stddev), ULP p50/p99
(float32 ULP of the reference), and the got/true ratio spread (median and p95/p5
over finite, non-negligible reference elements) — a tight cluster around a
non-1.0 constant would flag a scale/structural bug rather than rounding.

Also checks bitwise determinism (two calls on the same inputs), a prompt MUST rule.

Run with `-s` to see the table: scripts/run_safe_pytest.sh --dev <this file> -s
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose

from eval.golden_tests.mhc_pre.helpers import make_inputs, pytorch_mhc_pre, make_compute_config
from ttnn.bringup.mhc_pre import mhc_pre

_MIX = 24

SHAPES = [
    (1, 1, 64, 4 * 1024),  # small: C=1024
    (1, 100, 4 * 1024),  # ragged T, rank 3
    (1, 1, 640, 4 * 1792),  # DeepSeek-V4 TP4
    (1, 1, 640, 4 * 7168),  # DeepSeek-V4 full hidden (largest)
]


def _ulp_f32(ref):
    r = ref.to(torch.float32).abs()
    nxt = torch.nextafter(r, torch.full_like(r, float("inf")))
    return (nxt - r).to(torch.float64).clamp_min(torch.finfo(torch.float32).tiny)


def _metrics(got, ref):
    got = got.to(torch.float64).flatten()
    ref = ref.to(torch.float64).flatten()
    err = (got - ref).abs()
    rel_rms = (err.square().mean().sqrt() / ref.std()).item()
    ulp = err / _ulp_f32(ref)
    mask = torch.isfinite(ref) & (ref.abs() > 1e-3 * ref.abs().max())
    ratio = got[mask] / ref[mask]
    q = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return {
        "max_abs": err.max().item(),
        "mean_abs": err.mean().item(),
        "rel_rms": rel_rms,
        "ulp_p50": torch.quantile(ulp, 0.5).item(),
        "ulp_p99": torch.quantile(ulp, 0.99).item(),
        "ratio_med": q[1].item(),
        "ratio_p95_p5": (q[2] - q[0]).item(),
    }


@pytest.mark.parametrize("x_shape", SHAPES, ids=lambda s: "X" + "x".join(map(str, s)))
def test_mhc_pre_precision_baseline(device, x_shape):
    w_shape = (x_shape[-1], _MIX)
    x, w, b, scale = make_inputs(x_shape, w_shape, dtype=ttnn.float32, weight_dtype=ttnn.float32, seed=7)
    refs = pytorch_mhc_pre(x, w, b, scale=scale)

    def dev(t):
        return ttnn.from_torch(
            t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    tx, tw, tb = dev(x), dev(w), dev(b)
    outs = mhc_pre(tx, tw, tb, scale=scale, compute_kernel_config=make_compute_config())
    outs2 = mhc_pre(tx, tw, tb, scale=scale, compute_kernel_config=make_compute_config())

    for name, o, o2, ref in zip(("y", "post", "comb"), outs, outs2, refs):
        got = ttnn.to_torch(o)
        got2 = ttnn.to_torch(o2)
        assert torch.equal(got, got2), f"{name}: two calls on the same inputs are not bitwise identical"
        passing, pcc_msg = assert_with_pcc(ref, got, pcc=0.99999)
        _, allclose_msg = comp_allclose(ref, got)
        m = _metrics(got, ref)
        print(
            f"\nPRECISION {x_shape} {name:4s} {pcc_msg} | max_abs={m['max_abs']:.3e} mean_abs={m['mean_abs']:.3e} "
            f"rel_rms={m['rel_rms']:.3e} ulp_p50={m['ulp_p50']:.0f} ulp_p99={m['ulp_p99']:.0f} "
            f"ratio_med={m['ratio_med']:.6f} ratio_p95-p5={m['ratio_p95_p5']:.3e} | {allclose_msg}"
        )
        # Recommended tolerances (fp32 streams, tf32-class FPU projection / y-mix).
        assert m["rel_rms"] < 2e-3, f"{name}: rel_rms {m['rel_rms']}"
        # Scale-bug detector. fp32 X goes through the FPU with ~9 explicit mantissa bits (truncating), so y
        # carries a small known negative bias (ratio median ~0.99932 on BH, which matches a CPU emulation that
        # truncates 14 bits). A structural scale bug would be a tight cluster: spread << offset.
        offset = abs(m["ratio_med"] - 1.0)
        assert offset < 1.5e-3, f"{name}: ratio median {m['ratio_med']} (scale bug?)"
        assert m["ratio_p95_p5"] > 2 * offset, f"{name}: ratio tightly clustered off 1.0 -> scale bug"


# Refinement 1: bf16 X x fp32 W (the perf-focus contract). Before the exact W hi/lo split these shapes sat
# at post/comb rel-RMS 5.0-5.5e-4 vs the golden ("coeff", bfloat16) gate of 5e-4.
BF16_FP32W_SHAPES = [
    (1, 1, 17, 4 * 128),
    (1, 1, 1000, 4 * 7168),
    (1, 1, 256, 4 * 6144),
]


@pytest.mark.parametrize("x_shape", BF16_FP32W_SHAPES, ids=lambda s: "X" + "x".join(map(str, s)))
def test_mhc_pre_bf16_stream_fp32_weight_precision(device, x_shape):
    w_shape = (x_shape[-1], _MIX)
    x, w, b, scale = make_inputs(x_shape, w_shape, dtype=ttnn.bfloat16, weight_dtype=ttnn.float32, seed=7)
    refs = pytorch_mhc_pre(x, w, b, scale=scale)

    def dev(t, dt):
        return ttnn.from_torch(
            t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    outs = mhc_pre(
        dev(x, ttnn.bfloat16),
        dev(w, ttnn.float32),
        dev(b, ttnn.float32),
        scale=scale,
        compute_kernel_config=make_compute_config(),
    )
    for name, o, ref in zip(("y", "post", "comb"), outs, refs):
        m = _metrics(ttnn.to_torch(o), ref)
        print(f"\nPRECISION bf16X/fp32W {x_shape} {name:4s} rel_rms={m['rel_rms']:.3e} max_abs={m['max_abs']:.3e}")
        if name != "y":
            assert m["rel_rms"] < 5e-4, f"{name}: rel_rms {m['rel_rms']}"


# Refinement 2: fp32 streams use the exact-grid split, so the mix is ~fp32-accurate. The post logits
# z = mix * r + b (recovered from post = 2 sigmoid(z)) must match the fp64 reference to well below the
# ~2.9e-4 rms the FPU's tf32 read + in-tile rounding leaves (measured 3.5e-5 on T64 nC4096 a_res=30); a
# 1e-4 rms noise level is what the large-Sinkhorn-logit worst-row gate tolerates (CPU noise sweep).
FP32_EXACT_SHAPES = [(1, 1, 64, 4096), (1, 1, 640, 7168)]


@pytest.mark.parametrize("x_shape", FP32_EXACT_SHAPES, ids=lambda s: "X" + "x".join(map(str, s)))
def test_mhc_pre_fp32_stream_exact_projection(device, x_shape):
    nc = x_shape[-1]
    x, w, b, scale = make_inputs(
        x_shape, (nc, 24), dtype=ttnn.float32, weight_dtype=ttnn.float32, seed=42, logit_scale=30.0
    )
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    _, post_d, _ = mhc_pre(dev(x), dev(w), dev(b), scale=scale, compute_kernel_config=make_compute_config())
    p = ttnn.to_torch(post_d).double().reshape(-1, 4)
    X = x.double().reshape(-1, nc)
    r = torch.rsqrt(X.square().mean(-1, keepdim=True) + 1e-6)
    z_ref = (X @ w.double())[:, 4:8] * r + b.double().reshape(-1)[4:8]
    z_dev = torch.log((p / 2) / (1 - p / 2))
    z_rms = (z_dev - z_ref).pow(2).mean().sqrt().item()
    assert z_rms < 1e-4, f"fp32-stream post-logit rms error {z_rms:.3e} (exact-grid split expected ~3.5e-5)"
