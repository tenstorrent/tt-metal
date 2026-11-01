# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fuse_add_mean bake-off: the private op copy (ln_variant) under MEAN_VARIANT x perf rows.

Env:
  FAM_VARIANTS  comma list of MEAN_VARIANT values (default "0,1,2,3,4,5")
  FAM_CASES     comma list of case ids (default: all)
  FAM_PRELOAD   1 -> reader pushes only after all reads land (compute-only isolation)
Correctness (PCC / rel-RMS vs torch at the golden tolerance) is the only assert.
"""

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import ttnn

sys.path.insert(0, str(Path(__file__).parent))

import ln_variant  # noqa: E402
from ln_variant import layer_norm_tile_program_descriptor as pd  # noqa: E402

from eval.golden_tests.layer_norm_tile.feature_spec import LOOSE_CASES  # noqa: E402
from eval.golden_tests.layer_norm_tile.helpers import (  # noqa: E402
    _MEMORY_CONFIG,
    _TORCH_DTYPE,
    _tolerance_for,
    create_ttnn_input_tensor,
    make_affine,
    pytorch_layer_norm,
)

PERF_CASES = [case for case in LOOSE_CASES if "achievable_ns" in case.get("extras", {})]
CASES = dict(
    zip(
        [
            "focus1",
            "focus2_allDRAM_res",
            "focus3_allDRAM_nores",
            "allL1_res",
            "L1_nores",
            "8x384_allDRAM_res",
            "32x32_allDRAM_res",
            "affineTILE_DRAM",
        ],
        PERF_CASES,
    )
)
CASES["focus1_bf16dest"] = {**CASES["focus1"], "fp32_dest_acc_en": False}
CASES["focus3_bf16dest"] = {**CASES["focus3_allDRAM_nores"], "fp32_dest_acc_en": False}
CASES["hnonaligned_3x500_res"] = next(c for c in LOOSE_CASES if c["inputs"] == ((3, 1, 500, 768),))
CASES["hnonaligned_3x500_nores"] = {
    **CASES["hnonaligned_3x500_res"],
    "residual": "no_residual",
    "extras": {"output_buffer_type": ttnn.BufferType.DRAM},
}

VARIANTS = [int(v) for v in os.environ.get("FAM_VARIANTS", "0,1,2,3,4,5").split(",")]
CASE_IDS = os.environ.get("FAM_CASES", ",".join(CASES)).split(",")
PRELOAD = os.environ.get("FAM_PRELOAD", "0") == "1"


def _metrics(actual, expected):
    a = actual.to(torch.float64).flatten()
    e = expected.to(torch.float64).flatten()
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
    rel_rms = (torch.sqrt(torch.mean((a - e) ** 2)) / torch.std(e)).item()
    return pcc, rel_rms


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_variant(device, variant, case_id):
    case = CASES[case_id]
    pd.MEAN_VARIANT = variant
    pd.PRELOAD = PRELOAD
    shape = case["inputs"][0]
    W = shape[-1]
    extras = case.get("extras", {})
    dtype = case["dtype"]
    torch.manual_seed(0)
    x = torch.randn(shape).to(_TORCH_DTYPE[dtype])
    res = torch.randn(shape).to(_TORCH_DTYPE[dtype]) if case["residual"] == "residual" else None
    gamma = torch.randn(W).to(_TORCH_DTYPE[case["affine_dtype"]])
    beta = torch.randn(W).to(_TORCH_DTYPE[case["affine_dtype"]])
    expected = pytorch_layer_norm(x, gamma, beta, res)

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = case["fp32_dest_acc_en"]
    cfg.math_approx_mode = False

    tx = create_ttnn_input_tensor(x, device, dtype=dtype, layout=case["layout"], buffer_type=case["buffer_type"])
    tres = None
    if res is not None:
        tres = create_ttnn_input_tensor(
            res,
            device,
            dtype=dtype,
            layout=case["layout"],
            buffer_type=extras.get("residual_buffer_type", case["buffer_type"]),
        )
    affine_kw = dict(dtype=case["affine_dtype"], layout=case["affine_layout"], buffer_type=case["affine_buffer_type"])
    tg = make_affine(gamma, device, **affine_kw)
    tb = make_affine(beta, device, **affine_kw)
    out_bt = extras.get("output_buffer_type")
    out = ln_variant.layer_norm(
        tx,
        tg,
        tb,
        tres,
        compute_kernel_config=cfg,
        memory_config=_MEMORY_CONFIG[out_bt] if out_bt is not None else None,
    )
    actual = ttnn.to_torch(out)
    pcc, rel_rms = _metrics(actual, expected)
    pcc_min, rms_max = _tolerance_for(dtype, case["fp32_dest_acc_en"])
    print(f"\nFAM_RESULT variant={variant} case={case_id} preload={int(PRELOAD)} pcc={pcc:.6f} rel_rms={rel_rms:.5f}")
    assert pcc >= pcc_min and rel_rms <= rms_max, f"pcc {pcc} rel_rms {rel_rms}"
