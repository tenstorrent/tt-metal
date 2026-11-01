# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole-op normalize/affine bake-off: LN_NA=<variant> scripts/run_safe_pytest.sh [--profile] <this file>.

Rows = the 8 LOOSE_CASES perf rows + fp32_dest_acc_en=False + h_non_aligned (3,1,500,768). Correctness vs torch
at the golden tolerance; PCC / rel-RMS printed per row (NA_METRICS lines).
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import ttnn

sys.path.insert(0, str(Path(__file__).parent))

from eval.golden_tests.layer_norm_tile.feature_spec import LOOSE_CASES  # noqa: E402
from eval.golden_tests.layer_norm_tile.helpers import (  # noqa: E402
    _MEMORY_CONFIG,
    _TORCH_DTYPE,
    _tolerance_for,
    check_output,
    create_ttnn_input_tensor,
    make_affine,
    pytorch_layer_norm,
)
from install import install  # noqa: E402

from ttnn.operations.layer_norm_tile import layer_norm  # noqa: E402

VARIANT = install()

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
FOCUS = PERF_CASES[0]
CASES = dict(zip(PERF_IDS, PERF_CASES))
CASES["focus1_fp32dest_off"] = {**FOCUS, "fp32_dest_acc_en": False}
CASES["h_non_aligned_3x500"] = {**FOCUS, "inputs": ((3, 1, 500, 768),)}


def run_case(device, name, case):
    extras = case["extras"]
    shape = case["inputs"][0]
    W = shape[-1]
    dtype = case["dtype"]
    torch.manual_seed(0)
    x = torch.randn(shape).to(_TORCH_DTYPE[dtype])
    res = torch.randn(shape).to(_TORCH_DTYPE[dtype]) if case["residual"] == "residual" else None
    gamma = torch.randn(W).to(_TORCH_DTYPE[case["affine_dtype"]])
    beta = torch.randn(W).to(_TORCH_DTYPE[case["affine_dtype"]])
    expected = pytorch_layer_norm(x, gamma, beta, res)

    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = ttnn.MathFidelity.HiFi4
    config.fp32_dest_acc_en = case["fp32_dest_acc_en"]
    config.math_approx_mode = False

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
    affine = dict(dtype=case["affine_dtype"], layout=case["affine_layout"], buffer_type=case["affine_buffer_type"])
    tg = make_affine(gamma, device, **affine)
    tb = make_affine(beta, device, **affine)
    out_buf = extras.get("output_buffer_type")
    out = layer_norm(
        tx,
        tg,
        tb,
        tres,
        compute_kernel_config=config,
        memory_config=_MEMORY_CONFIG[out_buf] if out_buf is not None else None,
    )
    metrics = check_output(
        out,
        expected,
        shape=shape,
        dtype=dtype,
        expected_layout=case["layout"],
        tolerance=_tolerance_for(dtype, case["fp32_dest_acc_en"]),
    )
    print(f"NA_METRICS variant={VARIANT} case={name} pcc={metrics.pcc:.7f} rel_rms={metrics.rms:.6f}")


@pytest.mark.parametrize("name", list(CASES))
def test_variant(device, name):
    run_case(device, name, CASES[name])
