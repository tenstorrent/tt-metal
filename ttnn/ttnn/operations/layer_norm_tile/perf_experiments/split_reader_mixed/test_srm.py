# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""split_reader_mixed bake-off: whole op per (case, variant) through the variant/ descriptor copy.

SRM_CASES / SRM_VARIANTS (comma lists, env) select the grid; run order is appended to run_order.txt.
Correctness: golden tolerance vs torch and bit-exact vs the first variant of the case.
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

import swap  # noqa: E402
from ttnn.operations.layer_norm_tile import layer_norm  # noqa: E402

HERE = Path(__file__).parent
pytestmark = pytest.mark.use_module_device
_DRAM, _L1 = ttnn.BufferType.DRAM, ttnn.BufferType.L1
PERF = [c for c in LOOSE_CASES if "achievable_ns" in c.get("extras", {})]
CASES = dict(
    zip(
        ["focus1", "focus2", "focus3", "allL1_res", "L1_nores", "8x384", "32x32", "affTILE"],
        PERF,
    )
)
CASES["focus1_bf16dest"] = dict(CASES["focus1"], fp32_dest_acc_en=False)
CASES["3x500_inL1_resDRAM"] = next(c for c in LOOSE_CASES if c["inputs"][0] == (3, 1, 500, 768))
CASES["inDRAM_resL1_outL1"] = next(
    c for c in LOOSE_CASES if c["buffer_type"] == _DRAM and c.get("extras", {}).get("residual_buffer_type") == _L1
)
CASES["2x512_allL1_res"] = dict(CASES["allL1_res"], inputs=((2, 1, 512, 768),))
# 32-core-ish grid, mixed tiers (few active cores: the 2*active >= grid gate keeps it on the reader).
CASES["32x32_inL1_resDRAM"] = dict(CASES["focus1"], inputs=((32, 1, 32, 768),))

VARIANTS = {
    "base": dict(SRM_GATE="same_tier", SRM_PLAN="writer_wins", SRM_DUALNOC=False),
    "ow_writer_wins": dict(SRM_GATE="any", SRM_PLAN="writer_wins", SRM_DUALNOC=False),
    "ow_reader_wins": dict(SRM_GATE="any", SRM_PLAN="reader_wins", SRM_DUALNOC=False),
    "ow_fixed01": dict(SRM_GATE="any", SRM_PLAN="fixed01", SRM_DUALNOC=False),
    "ow_fixed10": dict(SRM_GATE="any", SRM_PLAN="fixed10", SRM_DUALNOC=False),
    "dualnoc": dict(SRM_GATE="same_tier", SRM_PLAN="writer_wins", SRM_DUALNOC=True),
    "ow_dram_y": dict(SRM_GATE="any", SRM_PLAN="dram_y", SRM_DUALNOC=False),
    "dual_dram_y": dict(SRM_GATE="same_tier", SRM_PLAN="dram_y", SRM_DUALNOC=True),
}
SEL_CASES = os.environ.get("SRM_CASES", "focus1").split(",")
SEL_VARIANTS = os.environ.get("SRM_VARIANTS", "base").split(",")
_first_out = {}


def _run(case, device):
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
    out = layer_norm(
        tx, tg, tb, tr, compute_kernel_config=cfg, memory_config=_MEMORY_CONFIG[obt] if obt is not None else None
    )
    got = ttnn.to_torch(out).float()
    exp = expected.float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), exp.flatten()]))[0, 1].item()
    return out, got, expected, pcc


@pytest.mark.parametrize("variant", SEL_VARIANTS)
@pytest.mark.parametrize("case_name", SEL_CASES)
def test_srm(device, case_name, variant):
    for key, value in VARIANTS[variant].items():
        setattr(swap.pd, key, value)
    case = CASES[case_name]
    seen = []
    orig = swap.pd.select_blocking

    def spy(*a, **k):
        blocking = orig(*a, **k)
        seen.append(blocking.residual_on_writer if blocking else None)
        return blocking

    swap.op_module.select_blocking = spy
    out, got, expected, pcc = _run(case, device)
    if pcc != pcc:  # precompile pass: placeholder output
        pytest.skip("precompile pass")
    with open(HERE / "run_order.txt", "a") as f:
        f.write(f"{case_name} {variant}\n")
    swap.op_module.select_blocking = orig
    print(f"SRM_RESULT {case_name} {variant} pcc={pcc:.6f} residual_on_writer={seen}")
    check_output(
        out,
        expected,
        shape=case["inputs"][0],
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=_tolerance_for(ttnn.bfloat16, case["fp32_dest_acc_en"]),
    )
    if case_name in _first_out:
        assert torch.equal(_first_out[case_name], got), "variant changed output bits"
    _first_out[case_name] = got
