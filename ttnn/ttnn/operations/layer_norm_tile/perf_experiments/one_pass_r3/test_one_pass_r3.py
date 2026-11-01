# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""one_pass_r3 bake-off: the private op copy (ln_variant) under knob variants x perf rows.

Env:
  OP_VARIANTS  comma list of VARIANTS keys (default: all), run variant-major
  OP_CASES     comma list of CASES keys (default: all)
Correctness (PCC / rel-RMS vs torch at the golden tolerance) is the only assert in test_perf; test_large_mean
ports test_layer_norm_tile_row_stats.py::test_large_mean (variant vs the copy's base two-pass).
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
        ["focus1", "focus2", "focus3", "allL1_res", "L1_nores", "8x384", "32x32", "affTILE"],
        PERF_CASES,
    )
)
CASES["g_focus1_bf16"] = {**CASES["focus1"], "fp32_dest_acc_en": False}
CASES["g_3x500"] = next(c for c in LOOSE_CASES if c["inputs"] == ((3, 1, 500, 768),))
CASES["g_64x17"] = dict(
    CASES["focus1"],
    inputs=((64, 1, 17, 768),),
    residual="no_residual",
    buffer_type=CASES["focus3"]["buffer_type"],
    affine_layout=CASES["affTILE"]["affine_layout"],
    extras={"output_buffer_type": CASES["focus3"]["buffer_type"]},
)
CASES["g_2x512"] = dict(CASES["allL1_res"], inputs=((2, 1, 512, 768),))

VARIANTS = {
    "base": {},
    "shift": {"ONE_PASS_ALL": True, "SHIFT_MODE": 0},
    "unshift": {"ONE_PASS_ALL": True, "SHIFT_MODE": 1},
    "sym": {"ONE_PASS_ALL": True, "SHIFT_MODE": 2},
    # balanced streamed rows only change their residual path (op's gates otherwise)
    "sym_gated": {"SHIFT_MODE": 2},
}

VARIANT_IDS = os.environ.get("OP_VARIANTS", ",".join(VARIANTS)).split(",")
CASE_IDS = os.environ.get("OP_CASES", ",".join(CASES)).split(",")


def _metrics(actual, expected):
    a = actual.to(torch.float64).flatten()
    e = expected.to(torch.float64).flatten()
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
    rel_rms = (torch.sqrt(torch.mean((a - e) ** 2)) / torch.std(e)).item()
    return pcc, rel_rms


def _apply(monkeypatch, variant):
    for knob, value in VARIANTS[variant].items():
        monkeypatch.setattr(pd, knob, value)


def _cfg(fp32_dest_acc_en):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
    cfg.math_approx_mode = False
    return cfg


def _run(device, case, x, res, gamma, beta):
    extras = case.get("extras", {})
    dtype = case["dtype"]
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
    out_bt = extras.get("output_buffer_type")
    out = ln_variant.layer_norm(
        tx,
        make_affine(gamma, device, **affine_kw),
        make_affine(beta, device, **affine_kw),
        tres,
        compute_kernel_config=_cfg(case["fp32_dest_acc_en"]),
        memory_config=_MEMORY_CONFIG[out_bt] if out_bt is not None else None,
    )
    return ttnn.to_torch(out)


def _inputs(case, mean=0.0):
    shape = case["inputs"][0]
    W = shape[-1]
    dt = _TORCH_DTYPE[case["dtype"]]
    torch.manual_seed(0)
    x = (mean + torch.randn(shape)).to(dt)
    res = torch.randn(shape).to(dt) if case["residual"] == "residual" else None
    gamma = torch.randn(W).to(_TORCH_DTYPE[case["affine_dtype"]])
    beta = torch.randn(W).to(_TORCH_DTYPE[case["affine_dtype"]])
    return x, res, gamma, beta


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("variant", VARIANT_IDS)
def test_perf(device, monkeypatch, variant, case_id):
    _apply(monkeypatch, variant)
    case = CASES[case_id]
    x, res, gamma, beta = _inputs(case)
    expected = pytorch_layer_norm(x, gamma, beta, res)
    actual = _run(device, case, x, res, gamma, beta)
    pcc, rel_rms = _metrics(actual, expected)
    pcc_min, rms_max = _tolerance_for(case["dtype"], case["fp32_dest_acc_en"])
    print(f"\nOP_RESULT variant={variant} case={case_id} pcc={pcc:.6f} rel_rms={rel_rms:.5f}")
    assert pcc >= pcc_min and rel_rms <= rms_max, f"pcc {pcc} rel_rms {rel_rms}"


# test_layer_norm_tile_row_stats.py::test_large_mean, on the copy: |mean| = 100, std 1. The variant may not be
# less precise than the copy's base (resident two-pass where the op gates it); printed for the option table.
LARGE_MEAN_SHAPES = {
    "unbalanced_8x512_L1": ((8, 1, 512, 768), ttnn.BufferType.L1),
    "balanced_11x320_DRAM": ((11, 1, 320, 768), ttnn.BufferType.DRAM),
    "unbalanced_8x512_DRAM": ((8, 1, 512, 768), ttnn.BufferType.DRAM),
}


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize("residual", ["residual", "no_residual"])
@pytest.mark.parametrize("shape_id", list(LARGE_MEAN_SHAPES))
@pytest.mark.parametrize("variant", [v for v in VARIANT_IDS if v != "base"])
def test_large_mean(device, monkeypatch, variant, shape_id, residual, fp32_dest_acc_en):
    shape, buf = LARGE_MEAN_SHAPES[shape_id]
    case = dict(
        CASES["focus1"],
        inputs=(shape,),
        residual=residual,
        buffer_type=buf,
        fp32_dest_acc_en=fp32_dest_acc_en,
        extras={"residual_buffer_type": buf, "output_buffer_type": buf},
    )
    x, res, gamma, beta = _inputs(case, mean=100.0)
    expected = pytorch_layer_norm(x, gamma, beta, res)
    base = _metrics(_run(device, case, x, res, gamma, beta), expected)
    with monkeypatch.context() as patch:
        _apply(patch, variant)
        pcc, rel_rms = _metrics(_run(device, case, x, res, gamma, beta), expected)
    print(
        f"\nOP_LARGE_MEAN variant={variant} shape={shape_id} {residual} fp32={int(fp32_dest_acc_en)} "
        f"pcc={pcc:.6f} rel_rms={rel_rms:.5f} base_pcc={base[0]:.6f} base_rel_rms={base[1]:.5f}"
    )
    assert pcc >= base[0] - 1e-5 and rel_rms <= 1.05 * base[1] + 1e-4, (pcc, rel_rms, base)


@pytest.mark.parametrize("variant", [v for v in VARIANT_IDS if v != "base"])
def test_constant_rows(device, monkeypatch, variant):
    """var = 0: the one-pass variance must clamp its rounding residue (output = beta, not NaN)."""
    _apply(monkeypatch, variant)
    case = dict(CASES["focus2"])
    shape = case["inputs"][0]
    x = torch.full(shape, 3.0).to(torch.bfloat16)
    res = torch.full(shape, 0.5).to(torch.bfloat16)
    torch.manual_seed(1)
    gamma = torch.randn(shape[-1]).to(torch.bfloat16)
    beta = torch.randn(shape[-1]).to(torch.bfloat16)
    actual = _run(device, case, x, res, gamma, beta).float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, beta.float().expand_as(actual), atol=0.05, rtol=0.02)
