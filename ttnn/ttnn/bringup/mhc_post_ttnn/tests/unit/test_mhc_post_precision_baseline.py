# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for mhc_post (verifier-owned; measures, then asserts a loose floor).

Per shape: PCC, max / mean abs error, relative RMS (to the reference stddev), ULP error of the fp32
output (relative to the output value, and to the term magnitude sum_k |term_k|), the signed relative bias (the residual-highway
shrink detector) and the got/true ratio spread (the uniform-scale-bug detector).

All four SUPPORTED (X dtype x F dtype) cells are measured. fp32 X' is gated on the exact-mix bounds;
bf16 X' on a single RNE output rounding (rel-RMS ~ 2^-9, unbiased). Run with `-s` to see the table.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.bringup.mhc_post_ttnn import mhc_post

N = 4

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}


def _sinkhorn(logits, iters=20, eps=1e-6):
    m = torch.softmax(logits, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


def _reference_f64(f, x, post, comb):
    n = post.shape[-1]
    lead, C = tuple(f.shape[:-1]), f.shape[-1]
    ff = f.double().reshape(-1, 1, C)
    xx = x.double().reshape(-1, n, C)
    pp = post.double().reshape(-1, n, 1)
    mm = comb.double().reshape(-1, n, n)
    return (pp * ff + torch.einsum("tij,tic->tjc", mm, xx)).reshape(*lead, n * C)


def _term_scale_f64(f, x, post, comb):
    """sum_k |term_k| per output element: the magnitude the fp32 roundings act on (cancellation-safe ULP base)."""
    n = post.shape[-1]
    lead, C = tuple(f.shape[:-1]), f.shape[-1]
    ff = f.double().abs().reshape(-1, 1, C)
    xx = x.double().abs().reshape(-1, n, C)
    pp = post.double().abs().reshape(-1, n, 1)
    mm = comb.double().abs().reshape(-1, n, n)
    return (pp * ff + torch.einsum("tij,tic->tjc", mm, xx)).reshape(*lead, n * C)


def _ulp_fp32(ref32):
    """Spacing of fp32 at each reference value."""
    return (torch.nextafter(ref32.abs(), torch.tensor(float("inf"))) - ref32.abs()).double()


SHAPES = [
    pytest.param((32, 4 * 32), id="T32_C32_single_tile"),
    pytest.param((1, 128, 4 * 1024), id="T128_C1024"),
    pytest.param((1, 100, 4 * 1024), id="T100_C1024_non_aligned"),
    pytest.param((1, 1, 640, 4 * 7168), id="T640_C7168_dsv4"),
]


@pytest.mark.parametrize(
    "dtypes",
    [
        pytest.param((ttnn.float32, ttnn.float32), id="X_fp32-F_fp32"),
        pytest.param((ttnn.bfloat16, ttnn.bfloat16), id="X_bf16-F_bf16"),
        pytest.param((ttnn.float32, ttnn.bfloat16), id="X_fp32-F_bf16"),
        pytest.param((ttnn.bfloat16, ttnn.float32), id="X_bf16-F_fp32"),
    ],
)
@pytest.mark.parametrize("x_shape", SHAPES)
def test_mhc_post_precision_baseline(device, x_shape, dtypes):
    dtype, sublayer_dtype = dtypes
    g = torch.Generator().manual_seed(1234)
    lead, nc = tuple(x_shape[:-1]), x_shape[-1]
    C = nc // N
    f = torch.randn(lead + (C,), generator=g).to(TORCH_DTYPE[sublayer_dtype])
    x = torch.randn(x_shape, generator=g).to(TORCH_DTYPE[dtype])
    post = (2.0 * torch.sigmoid(torch.randn(lead + (N,), generator=g))).float()
    comb = _sinkhorn(torch.randn(*lead, N, N, generator=g)).reshape(lead + (N * N,)).float().contiguous()

    def dev(t, dt):
        return ttnn.from_torch(
            t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    out = mhc_post(dev(f, sublayer_dtype), dev(x, dtype), dev(post, ttnn.float32), dev(comb, ttnn.float32))
    got = ttnn.to_torch(out).to(torch.float32)

    ref64 = _reference_f64(f, x, post, comb)
    ref32 = ref64.to(torch.float32)

    pcc = torch.corrcoef(torch.stack([got.double().flatten(), ref64.flatten()]))[0, 1].item()
    err = got.double() - ref64
    max_abs = err.abs().max().item()
    mean_abs = err.abs().mean().item()
    rel_rms = (err.pow(2).mean().sqrt() / ref64.std()).item()
    ulp = (got.double() - ref32.double()).abs() / _ulp_fp32(ref32)
    ulp_p50, ulp_p99 = torch.quantile(ulp.flatten()[: 1 << 24].float(), torch.tensor([0.5, 0.99])).tolist()
    # ULP in units of the term magnitude (sum_k |term_k|): immune to cancellation in the sum.
    scale32 = _term_scale_f64(f, x, post, comb).to(torch.float32)
    ulp_s = (got.double() - ref64).abs() / _ulp_fp32(scale32)
    ulp_s_max, ulp_s_p99 = ulp_s.max().item(), torch.quantile(ulp_s.flatten()[: 1 << 24].float(), 0.99).item()
    d = err * ref64.sign()
    bias = (d.mean() / ref64.abs().mean()).item()
    bias_se = (d.std() / d.numel() ** 0.5 / ref64.abs().mean()).item()

    mask = ref64.abs() > 1e-3
    ratio = got.double()[mask] / ref64[mask]
    r_med = ratio.median().item()
    r_p5, r_p95 = torch.quantile(ratio[: 1 << 24].float(), torch.tensor([0.05, 0.95])).tolist()

    _, allclose_msg = comp_allclose(ref32, got)
    print(
        f"\nPRECISION {x_shape} X={dtype} F={sublayer_dtype}: pcc={pcc:.10f} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"rel_rms={rel_rms:.3e} ulp_out p50={ulp_p50:.1f} p99={ulp_p99:.1f} ulp_scale max={ulp_s_max:.2f} p99={ulp_s_p99:.2f} bias={bias:+.2e}(se {bias_se:.1e}) "
        f"ratio median={r_med:.9f} p5={r_p5:.9f} p95={r_p95:.9f} | {allclose_msg}"
    )

    assert torch.isfinite(got).all()
    if dtype == ttnn.float32:
        # fp32 streams: the SFPU mix is exact up to a few fp32 roundings (1 mul + n MADs); a bf16 F input
        # is exact in fp32, so the reference (built from the rounded F) holds the same bound.
        assert_with_pcc(ref32, got, pcc=0.9999999)
        assert rel_rms < 2e-6
        assert ulp_s_max <= 8
        assert abs(bias) <= 1e-6 + 6 * bias_se
        assert abs(r_med - 1.0) < 1e-6
    else:
        # bf16 X': one fp32 -> bf16 output rounding (RNE, unbiased) on top of the exact fp32 mix.
        assert_with_pcc(ref32, got, pcc=0.9999)
        assert rel_rms < 4e-3
        assert abs(bias) <= 1e-5 + 6 * bias_se
        assert abs(r_med - 1.0) < 1e-3
