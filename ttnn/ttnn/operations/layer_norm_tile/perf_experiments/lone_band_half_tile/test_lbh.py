# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""lone_band_half_tile bench: perf + guard rows through lbh_descriptor (load with -p lbh_swap).

Correctness gate per row = the golden tolerance (fp32 DEST: PCC 0.999, rel-RMS 0.02; bf16 DEST: 0.998, 0.05);
PCC and rel-RMS vs torch are printed as LBH_PRECISION lines. LBH_ROWS=name,name selects rows."""

import os

import pytest
import importlib

torch = importlib.import_module("torch")  # no global torch import under ttnn (pre-commit)
import ttnn

import lbh_descriptor
from eval.golden_tests.layer_norm_tile.helpers import (
    _MEMORY_CONFIG,
    _tolerance_for,
    create_ttnn_input_tensor,
    make_affine,
    pytorch_layer_norm,
)
from models.common.metrics import compute_pcc
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_perf_bench import GUARD_CASES, IDS, PERF_CASES
from ttnn.operations.layer_norm_tile import layer_norm

ROWS = {name: case for name, case in zip(IDS, PERF_CASES)}
ROWS.update(GUARD_CASES)
_ALL = dict(ROWS)
SEL = os.environ.get("LBH_ROWS")
if SEL:
    ROWS = {name: ROWS[name] for name in SEL.split(",")}


def run_case(device, case):
    shape = case["inputs"][0]
    extras = case.get("extras", {})
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    res = torch.randn(shape).to(torch.bfloat16) if case["residual"] == "residual" else None
    gamma = torch.randn(shape[-1]).to(torch.bfloat16)
    beta = torch.randn(shape[-1]).to(torch.bfloat16)
    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
    config.fp32_dest_acc_en = case["fp32_dest_acc_en"]
    config.math_approx_mode = False
    buf = case["buffer_type"]
    out_buf = extras.get("output_buffer_type")
    out = layer_norm(
        create_ttnn_input_tensor(x, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, buffer_type=buf),
        make_affine(
            gamma, device, dtype=ttnn.bfloat16, layout=case["affine_layout"], buffer_type=case["affine_buffer_type"]
        ),
        make_affine(
            beta, device, dtype=ttnn.bfloat16, layout=case["affine_layout"], buffer_type=case["affine_buffer_type"]
        ),
        create_ttnn_input_tensor(
            res,
            device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            buffer_type=extras.get("residual_buffer_type", buf),
        )
        if res is not None
        else None,
        compute_kernel_config=config,
        memory_config=_MEMORY_CONFIG[out_buf] if out_buf is not None else None,
    )
    expected = pytorch_layer_norm(x, gamma, beta, res).float()
    actual = ttnn.to_torch(out).float()
    pcc = compute_pcc(actual, expected)
    rel_rms = (torch.sqrt(torch.mean((actual - expected) ** 2)) / expected.std()).item()
    return pcc, rel_rms


@pytest.mark.parametrize("name", list(ROWS))
def test_lbh(device, name):
    case = ROWS[name]
    pcc, rel_rms = run_case(device, case)
    print(f"LBH_PRECISION {os.environ.get('LBH_VARIANT', 'half')} {name} pcc={pcc:.6f} rel_rms={rel_rms:.5f}")
    pcc_floor, rms_ceil = _tolerance_for(ttnn.bfloat16, case["fp32_dest_acc_en"])
    assert pcc >= pcc_floor and rel_rms <= rms_ceil, (pcc, rel_rms)


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize(
    "name", ["focus1_inL1_resDRAM_outDRAM", "focus3_allDRAM_nores", "L1_nores", "2x512_allL1_res", "32x32_allDRAM_res"]
)
def test_large_mean_precision(device, monkeypatch, name, fp32_dest_acc_en):
    """|mean| = 100: print precision per variant (compare LBH_VARIANT=base vs half); no gate, the bar is the base."""
    case = dict(_ALL[name], fp32_dest_acc_en=fp32_dest_acc_en)
    real_randn = torch.randn
    calls = []

    def shifted_randn(*args, **kwargs):
        out = real_randn(*args, **kwargs)
        calls.append(1)
        return out + 100.0 if len(calls) == 1 else out  # x only

    monkeypatch.setattr(torch, "randn", shifted_randn)
    pcc, rel_rms = run_case(device, case)
    print(
        f"LBH_LARGE_MEAN {os.environ.get('LBH_VARIANT', 'half')} {name} fp32={fp32_dest_acc_en} pcc={pcc:.6f} rel_rms={rel_rms:.5f}"
    )
