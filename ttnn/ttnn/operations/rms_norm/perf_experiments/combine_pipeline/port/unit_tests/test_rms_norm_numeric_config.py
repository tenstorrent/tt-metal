# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 1 — numerical configurability: bfloat8_b input / gamma and fp32_dest_acc_en=False.

Pins:
  * 16-bit DEST (fp32_dest_acc_en=False) on bfloat16 / bfloat8_b inputs, every blocking regime
    (k == 1 row-parallel, k > 1 W-split whose gathered partials now travel as bfloat16 tiles,
    multi-block depth-2), HiFi4 and the perf-profile HiFi2.
  * bfloat8_b TILE gamma (format-aware row-0 expansion: shared-exponent section re-patterned).
  * bfloat8_b refusals mirrored from feature_spec INVALID (ROW_MAJOR, non-tile-aligned).
"""

import pytest
import ttnn

from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.rms_norm import rms_norm


def torch_rms_norm(x, gamma=None, epsilon=1e-6):
    import torch  # function-local: the ttnn tree forbids global torch imports

    xf = x.to(torch.float32)
    y = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        y = y * gamma.to(torch.float32).reshape(-1)
    return y


def pcc(actual, expected):
    import torch  # function-local: the ttnn tree forbids global torch imports

    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    if torch.equal(a, e):
        return 1.0
    return torch.corrcoef(torch.stack([a, e]))[0, 1].item()


def config(fp32_dest, fidelity=ttnn.MathFidelity.HiFi4):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fidelity
    c.fp32_dest_acc_en = fp32_dest
    c.math_approx_mode = False
    return c


def run(device, shape, dtype, *, fp32_dest, fidelity=ttnn.MathFidelity.HiFi4, gamma_dtype=None, pcc_min=0.999):
    import torch  # function-local: the ttnn tree forbids global torch imports

    torch.manual_seed(7)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    W = shape[-1]
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    x_ref = ttnn.to_torch(tt_x).to(torch.float32)  # the quantized input the device actually sees
    g_ref, tt_g = None, None
    if gamma_dtype is not None:
        g = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        tt_g = ttnn.from_torch(g, dtype=gamma_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        g_ref = ttnn.to_torch(tt_g).to(torch.float32).reshape(-1)[:W]
    out = rms_norm(tt_x, gamma=tt_g, compute_kernel_config=config(fp32_dest, fidelity))
    assert out.dtype == dtype
    actual = ttnn.to_torch(out).to(torch.float32)
    expected = torch_rms_norm(x_ref, g_ref)
    p = pcc(actual, expected)
    assert p >= pcc_min, f"PCC {p:.6f} < {pcc_min} shape={shape} dtype={dtype} fp32_dest={fp32_dest}"


SHAPES = [
    pytest.param((1, 1, 32, 64), id="small_k1"),
    pytest.param((2, 1, 1024, 256), id="R1_row_parallel"),
    pytest.param((1, 1, 32, 4096), id="R2_occupancy"),
    pytest.param((1, 1, 32, 16384), id="R2_fit"),
]


@pytest.mark.parametrize("gamma_dtype", [None, ttnn.bfloat16, ttnn.bfloat8_b], ids=["no_gamma", "g_bf16", "g_bf8b"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8b"])
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_16bit_dest(device, shape, dtype, gamma_dtype):
    run(
        device,
        shape,
        dtype,
        fp32_dest=False,
        gamma_dtype=gamma_dtype,
        pcc_min=0.995 if dtype == ttnn.bfloat8_b else 0.999,
    )


@pytest.mark.parametrize("gamma_dtype", [None, ttnn.bfloat8_b, ttnn.float32], ids=["no_gamma", "g_bf8b", "g_fp32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_bf8b_fp32_dest(device, shape, gamma_dtype):
    run(device, shape, ttnn.bfloat8_b, fp32_dest=True, gamma_dtype=gamma_dtype, pcc_min=0.995)


def test_rms_norm_bf16_gamma_bf8b_fp32_dest(device):
    run(device, (1, 1, 64, 256), ttnn.bfloat16, fp32_dest=True, gamma_dtype=ttnn.bfloat8_b, pcc_min=0.999)


# The Refinement 3 perf-profile config: bf16, 16-bit DEST, HiFi2, TILE gamma, soft PCC 0.9995.
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 7168), id="decode_7168"),
        pytest.param((1, 1, 640, 6144), id="640x6144"),
        pytest.param((1, 1, 6144, 12288), id="R2_fit_multi_block"),
    ],
)
def test_rms_norm_perf_profile_config(device, shape):
    run(
        device,
        shape,
        ttnn.bfloat16,
        fp32_dest=False,
        fidelity=ttnn.MathFidelity.HiFi2,
        gamma_dtype=ttnn.bfloat16,
        pcc_min=0.9995,
    )


def test_rms_norm_rejects_bf8b_non_aligned(device, expect_error):
    import torch  # function-local: the ttnn tree forbids global torch imports

    x = ttnn.from_torch(torch.randn(1, 1, 32, 50), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(ExcludedCell, r".*"):
        rms_norm(x)
