# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""stats_fusion whole-op A/B: LN_STATS_VARIANT=<0..3> scripts/run_safe_pytest.sh [--profile] <this file>.

Points the op's KERNEL_DIR at op_kernels/v<N> (reader/writer symlinked to the real op, compute =
the real compute with the variance stage swapped for stats_variants.hpp variant N). One variant per
pytest process. Correctness vs torch at the golden tolerance on every row.
"""

import os
from pathlib import Path

import pytest

from eval.golden_tests.layer_norm_tile.feature_spec import LOOSE_CASES
from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm
from ttnn.operations.layer_norm_tile import layer_norm_tile_program_descriptor as _pd

# LN_STATS_VARIANT=<0..5> -> op_kernels/v<N>; "grad" -> graduate/ (the graduation-ready kernel).
VARIANT = os.environ.get("LN_STATS_VARIANT", "0")
_pd.KERNEL_DIR = Path(__file__).parent / ("graduate" if VARIANT == "grad" else f"op_kernels/v{VARIANT}")

PERF_CASES = [case for case in LOOSE_CASES if "achievable_ns" in case.get("extras", {})]
PERF_IDS = [
    "focus1_inL1_resDRAM_outDRAM",
    "focus2_allDRAM_res",
    "focus3_allDRAM_nores",
    "allL1_res",
    "L1_nores",
    "8x384_allDRAM_res",
    "32x32_allDRAM_res",
    "affineTILE_DRAM",
]


def _with(case, **kw):
    out = dict(case)
    out["extras"] = dict(case["extras"])
    out.update(kw)
    return out


FOCUS = PERF_CASES[0]
EXTRA_CASES = {
    "focus1_fp32dest_off": _with(FOCUS, fp32_dest_acc_en=False),
    "h_non_aligned_3x500": _with(FOCUS, inputs=((3, 1, 500, 768),)),
}


@pytest.mark.parametrize("case", PERF_CASES, ids=PERF_IDS)
def test_perf(device, case):
    case = dict(case)
    inputs = case.pop("inputs")
    run_layer_norm(inputs, device=device, **case)


@pytest.mark.parametrize("name", list(EXTRA_CASES))
def test_extra(device, name):
    case = dict(EXTRA_CASES[name])
    inputs = case.pop("inputs")
    run_layer_norm(inputs, device=device, **case)


@pytest.mark.parametrize("offset", [0.0, 100.0], ids=["zero_mean", "mean100"])
@pytest.mark.parametrize("fp32_dest", [True, False], ids=["fp32dest", "bf16dest"])
def test_precision(device, offset, fp32_dest):
    """Focus shape metrics vs fp32 torch (output rel-RMS / PCC), printed for the option menu."""
    import torch
    import ttnn
    from eval.metrics import compute_metrics_torch
    from ttnn.operations.layer_norm_tile import layer_norm

    torch.manual_seed(0)
    shape = (8, 1, 512, 768)
    x = (torch.randn(shape) * 2.0 + offset).to(torch.bfloat16)
    res = torch.randn(shape).to(torch.bfloat16)
    gamma = torch.randn(768).to(torch.bfloat16)
    beta = torch.randn(768).to(torch.bfloat16)
    z = x.float() + res.float()
    expected = torch.nn.functional.layer_norm(z.double(), (768,), gamma.double(), beta.double(), eps=1e-5)

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest
    cfg.math_approx_mode = False
    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tr = ttnn.from_torch(res, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tg = ttnn.from_torch(
        gamma.reshape(1, 1, 24, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tb = ttnn.from_torch(
        beta.reshape(1, 1, 24, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    out = layer_norm(tx, tg, tb, tr, epsilon=1e-5, compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    actual = ttnn.to_torch(out).float()
    m = compute_metrics_torch(actual, expected.float(), readback_dtype=torch.bfloat16)
    print(f"\nSTATS_PREC variant={VARIANT} fp32_dest={fp32_dest} offset={offset} pcc={m.pcc:.6f} rel_rms={m.rms:.6f}")
    assert m.pcc > 0.99
