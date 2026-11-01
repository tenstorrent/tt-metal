# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""lone_band_fold domain: correctness + precision of R3 lone-band rows beyond the bench rows (load with -p lbf_swap).

Prints "LBF_DOMAIN <id> fold=<0|1> pcc=.. rel_rms=..". Run once with LBF_FOLD=0 and once with LBF_FOLD=1 to compare.
"""

import pytest
import importlib

torch = importlib.import_module("torch")  # no global torch import under ttnn (pre-commit)
import ttnn

import lbf_descriptor
from models.common.metrics import compute_pcc
from ttnn.operations.layer_norm_tile import layer_norm
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_layer_norm_tile import (
    DRAM,
    L1,
    compute_config,
    make_affine,
    reference_layer_norm,
    to_device,
)

HIFI4, LOFI = ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.LoFi
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

# (shape, input/output buf, residual buf or None, fp32_dest, fidelity, affine layout, mean)
CASES = {
    "8x512_resDRAM_mean100_fp32": ((8, 1, 512, 768), L1, DRAM, True, HIFI4, RM, 100.0),
    "8x512_resDRAM_mean100_bf16": ((8, 1, 512, 768), L1, DRAM, False, HIFI4, RM, 100.0),
    "32x32_resDRAM_mean100_fp32": ((32, 1, 32, 768), DRAM, DRAM, True, HIFI4, RM, 100.0),
    "32x32_resDRAM_mean100_bf16": ((32, 1, 32, 768), DRAM, DRAM, False, HIFI4, RM, 100.0),
    "2x512_L1_nores_mean100_fp32": ((2, 1, 512, 768), L1, None, True, HIFI4, RM, 100.0),
    "8x512_W64_res": ((8, 1, 512, 64), DRAM, DRAM, True, HIFI4, RM, 0.0),
    "8x512_W192_res": ((8, 1, 512, 192), DRAM, DRAM, True, HIFI4, RM, 0.0),
    "8x512_W1024_res": ((8, 1, 512, 1024), DRAM, DRAM, True, HIFI4, RM, 0.0),
    "8x512_W1536_nores": ((8, 1, 512, 1536), L1, None, True, HIFI4, RM, 0.0),
    "32x32_W1024_res": ((32, 1, 32, 1024), DRAM, DRAM, True, HIFI4, RM, 0.0),
    "32x32_W96_odd_res": ((32, 1, 32, 96), DRAM, DRAM, True, HIFI4, RM, 0.0),
    "8x512_W96_odd_res": ((8, 1, 512, 96), DRAM, DRAM, True, HIFI4, RM, 0.0),
    "32x32_LoFi_res": ((32, 1, 32, 768), DRAM, DRAM, True, LOFI, RM, 0.0),
    "8x512_LoFi_res": ((8, 1, 512, 768), L1, DRAM, True, LOFI, RM, 0.0),
    "32x32_affTILE_res": ((32, 1, 32, 768), DRAM, DRAM, True, HIFI4, TILE, 0.0),
    "8x512_affTILE_bf16": ((8, 1, 512, 768), DRAM, None, False, HIFI4, TILE, 0.0),
    "3x500_resL1": ((3, 1, 500, 768), L1, L1, True, HIFI4, RM, 0.0),
    "7x3200_multiblock_res": ((7, 1, 3200, 768), DRAM, DRAM, True, HIFI4, RM, 0.0),
    "5x1100_skip_band_nores": ((5, 1, 1100, 768), L1, None, False, HIFI4, RM, 0.0),
}
GATE = {True: (0.999, 0.02), False: (0.998, 0.05)}


@pytest.mark.parametrize("case", list(CASES.values()), ids=list(CASES))
def test_domain(device, case, request):
    shape, buf, res_buf, fp32_dest, fidelity, affine_layout, mean = case
    torch.manual_seed(0)
    x = (mean + torch.randn(shape)).to(torch.bfloat16)
    res = torch.randn(shape).to(torch.bfloat16) if res_buf is not None else None
    W = shape[-1]
    gamma = torch.randn(W).to(torch.bfloat16)
    beta = torch.randn(W).to(torch.bfloat16)
    out = layer_norm(
        to_device(x, device, buf),
        make_affine(gamma, device, affine_layout, L1),
        make_affine(beta, device, affine_layout, L1),
        to_device(res, device, res_buf) if res is not None else None,
        compute_kernel_config=compute_config(fp32_dest, fidelity),
    )
    expected = reference_layer_norm(x, gamma, beta, res)
    actual = ttnn.to_torch(out).float()
    assert torch.isfinite(actual).all()
    rel = (torch.sqrt(torch.mean((actual - expected) ** 2)) / expected.std()).item()
    pcc = compute_pcc(actual, expected)
    print(
        f"\nLBF_DOMAIN {request.node.callspec.id} fold={int(lbf_descriptor.FOLD_LONE_BAND)} pcc={pcc:.6f} rel_rms={rel:.5f}"
    )
    if mean == 0.0 and fidelity == HIFI4:
        pcc_floor, rel_ceil = GATE[fp32_dest]
        assert pcc >= pcc_floor and rel <= rel_ceil, (pcc, rel)
    else:
        assert pcc >= 0.99, (pcc, rel)
