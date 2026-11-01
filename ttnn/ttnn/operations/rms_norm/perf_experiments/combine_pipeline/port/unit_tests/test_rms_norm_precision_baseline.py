# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""rms_norm precision baseline (verifier, Phase 0).

Measures PCC, max / mean absolute error, relative RMS error, ULP error (in units of the OUTPUT dtype's
ULP at the reference value) and the got/true ratio spread (scale-bug detector) over a small shape set
that spans the blocking regimes:

  * (1,1,32,32)      — 1 tile, R1 row-parallel
  * (1,1,256,1024)   — 8 tile-rows x 32 W tiles, R1 (or R2 by occupancy on a large grid)
  * (2,1,512,2048)   — 32 tile-rows x 64 W tiles, R1 multi-core
  * (1,1,32,16384)   — 1 tile-row, R2 W split (cross-Tensix-core combine)

Phase 0 contract: fp32_dest_acc_en=True, HiFi4. Results are printed (run with -s) and asserted only
against the acceptance PCC floors — this file records the baseline, it does not tighten the contract.
"""

import pytest
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.rms_norm import rms_norm

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
PCC_FLOOR = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}

SHAPES = [(1, 1, 32, 32), (1, 1, 256, 1024), (2, 1, 512, 2048), (1, 1, 32, 16384)]


def _config():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c


def _reference(x, g, eps):
    import torch  # function-local: the ttnn tree forbids global torch imports

    xf = x.to(torch.float64)
    y = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    return y * g.to(torch.float64).reshape(-1)


def _ulp(ref, torch_dtype):
    """ULP of `torch_dtype` at each |ref| (float64 result)."""
    import torch  # function-local: the ttnn tree forbids global torch imports

    finfo = torch.finfo(torch_dtype)
    mant_bits = {torch.bfloat16: 7, torch.float32: 23}[torch_dtype]
    a = ref.abs().clamp(min=finfo.tiny)
    return torch.pow(2.0, torch.floor(torch.log2(a)) - mant_bits)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rms_norm_precision_baseline(device, shape, dtype, layout):
    import torch  # function-local: the ttnn tree forbids global torch imports

    torch.manual_seed(1234)
    tdt = TORCH_DTYPE[dtype]
    eps = 1e-6
    x = torch.randn(shape, dtype=torch.float32).to(tdt)
    g = torch.randn(shape[-1], dtype=torch.float32).to(tdt)
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    out = ttnn.to_torch(rms_norm(tx, gamma=tg, epsilon=eps, compute_kernel_config=_config()))
    actual = out.to(torch.float64)
    expected = _reference(x, g, eps)

    passed, pcc_msg = assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC_FLOOR[dtype])
    _, allclose_msg = comp_allclose(expected.to(torch.float32), actual.to(torch.float32))

    err = (actual - expected).abs()
    rel_rms = (torch.sqrt(torch.mean((actual - expected) ** 2)) / torch.sqrt(torch.mean(expected**2))).item()
    ulp_err = err / _ulp(expected, tdt)
    nz = expected.abs() > 1e-3
    ratio = actual[nz] / expected[nz]
    q = torch.quantile(ratio.float(), torch.tensor([0.05, 0.5, 0.95]))

    print(
        f"\nPRECISION {'x'.join(map(str, shape))} {dtype} {layout}: pcc={pcc_msg} "
        f"max_abs={err.max().item():.5g} mean_abs={err.mean().item():.5g} rel_rms={rel_rms:.5g} "
        f"ulp_p50={ulp_err.median().item():.3g} ulp_p99={torch.quantile(ulp_err.flatten()[::max(1, ulp_err.numel() // 1_000_000)].float(), 0.99).item():.3g} "
        f"ulp_max={ulp_err.max().item():.3g} ratio_p5/p50/p95={q[0].item():.5f}/{q[1].item():.5f}/{q[2].item():.5f} "
        f"| {allclose_msg}"
    )
    assert torch.isfinite(actual).all()
    assert passed, pcc_msg
