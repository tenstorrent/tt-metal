# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""noc_balance bake-off: one layer_norm call per (case, plan); correctness vs torch at the golden tolerance
and bit-exact vs the base plan (NoC choice must not change a single output bit).

Select plans with NB_PLANS=base,rd1_all,... and cases with NB_CASES=focus1,... (env).
Device ns: run under scripts/run_safe_pytest.sh --profile; order is written to run_order.txt.
"""

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import ttnn

sys.path.insert(0, str(Path(__file__).parent))

from eval.golden_tests.layer_norm_tile.feature_spec import LOOSE_CASES  # noqa: E402
from eval.golden_tests.layer_norm_tile.helpers import (  # noqa: E402
    _MEMORY_CONFIG,
    _tolerance_for,
    check_output,
    create_ttnn_input_tensor,
    make_affine,
    pytorch_layer_norm,
)

import ln_noc  # noqa: E402
from plans import PLANS  # noqa: E402

HERE = Path(__file__).parent
pytestmark = pytest.mark.use_module_device
PERF = [c for c in LOOSE_CASES if "achievable_ns" in c.get("extras", {})]
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
        PERF,
    )
)
CASES["focus1_bf16dest"] = dict(CASES["focus1"], fp32_dest_acc_en=False)
CASES["h_non_aligned"] = next(c for c in LOOSE_CASES if c["inputs"][0] == (3, 1, 500, 768))

# 200 tile-rows: 90 cores x 2 rows + 20 x 1 (heavy group is the majority), focus1 placement.
CASES["rows200_focus1"] = dict(CASES["focus1"], inputs=((25, 1, 256, 768),))
SEL_CASES = os.environ.get("NB_CASES", "focus1").split(",")
SEL_PLANS = os.environ.get("NB_PLANS", "base").split(",")
_baseline_out = {}


def _run(case, plan_name, device):
    extras = case.get("extras", {})
    shape = case["inputs"][0]
    W = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    r = torch.randn(shape).to(torch.bfloat16) if case["residual"] == "residual" else None
    g = torch.randn(W).to(torch.bfloat16)
    b = torch.randn(W).to(torch.bfloat16)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = case["fp32_dest_acc_en"]
    cfg.math_approx_mode = False
    expected = pytorch_layer_norm(x, g, b, r)
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tx = create_ttnn_input_tensor(x, device, buffer_type=case["buffer_type"], **kw)
    tr = (
        create_ttnn_input_tensor(r, device, buffer_type=extras.get("residual_buffer_type", case["buffer_type"]), **kw)
        if r is not None
        else None
    )
    akw = dict(dtype=ttnn.bfloat16, layout=case["affine_layout"], buffer_type=case["affine_buffer_type"])
    tg = make_affine(g, device, **akw)
    tb = make_affine(b, device, **akw)
    obt = extras.get("output_buffer_type")
    out = ln_noc.layer_norm(
        tx,
        tg,
        tb,
        tr,
        compute_kernel_config=cfg,
        memory_config=_MEMORY_CONFIG[obt] if obt is not None else None,
        plan=PLANS[plan_name],
    )
    got = ttnn.to_torch(out).float()
    exp = expected.float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), exp.flatten()]))[0, 1].item()
    rel_rms = ((got - exp).pow(2).mean().sqrt() / exp.std()).item()
    return out, got, expected, pcc, rel_rms


@pytest.mark.parametrize("plan_name", SEL_PLANS)
@pytest.mark.parametrize("case_name", SEL_CASES)
def test_noc_balance(device, case_name, plan_name):
    case = CASES[case_name]
    out, got, expected, pcc, rel_rms = _run(case, plan_name, device)
    if pcc != pcc:  # precompile pass: placeholder output, nothing to check
        pytest.skip("precompile pass")
    with open(HERE / "run_order.txt", "a") as f:
        f.write(f"{case_name} {plan_name}\n")
    print(f"NB_RESULT {case_name} {plan_name} pcc={pcc:.6f} rel_rms={rel_rms:.5f}")
    check_output(
        out,
        expected,
        shape=case["inputs"][0],
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=_tolerance_for(ttnn.bfloat16, case["fp32_dest_acc_en"]),
    )
    # Every plan of a case must be bit-identical (NoC choice moves bytes, never changes them).
    if case_name in _baseline_out:
        assert torch.equal(_baseline_out[case_name], got), "NoC plan changed output bits"
    _baseline_out[case_name] = got
