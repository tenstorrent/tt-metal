# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""lone_band_fold bench: test_perf_bench.py rows through this dir's descriptor copy (load with -p lbf_swap).

Each row runs helpers.run_layer_norm (the golden gate) and prints "LBF_PRECISION <row> pcc=.. rel_rms=..".
LBF_ROWS=a,b selects rows; LBF_FOLD=0|1 selects base / fold.
"""

import os

import pytest
import importlib

torch = importlib.import_module("torch")  # no global torch import under ttnn (pre-commit)
import ttnn

import lbf_descriptor
from models.common.metrics import compute_pcc
from eval.golden_tests.layer_norm_tile import helpers
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_perf_bench import GUARD_CASES, IDS, PERF_CASES

SHORT = ["focus1", "focus2", "focus3", "allL1_res", "L1_nores", "8x384", "32x32", "affTILE"]
SHORT += ["g_bf16dest", "g_3x500", "g_64x17", "g_2x512"]
ROWS = dict(zip(SHORT, list(PERF_CASES) + list(GUARD_CASES.values())))
SEL = os.environ.get("LBF_ROWS")
if SEL:
    ROWS = {name: ROWS[name] for name in SEL.split(",")}


@pytest.fixture
def capture_output(monkeypatch):
    """Keep the op output and reference that run_layer_norm hands to check_output."""
    seen = {}
    real = helpers.check_output

    def spy(actual, expected, **kwargs):
        seen["actual"] = ttnn.to_torch(actual).float()
        seen["expected"] = expected.float()
        return real(actual, expected, **kwargs)

    monkeypatch.setattr(helpers, "check_output", spy)
    return seen


@pytest.mark.parametrize("case", list(ROWS.values()), ids=list(ROWS))
def test_lbf(device, case, capture_output, request):
    assert lbf_descriptor.FOLD_LONE_BAND == (os.environ.get("LBF_FOLD", "1") != "0")
    case = dict(case)
    inputs = case.pop("inputs")
    try:
        helpers.run_layer_norm(inputs, device=device, **case)
    finally:
        if "actual" in capture_output:
            actual, expected = capture_output["actual"], capture_output["expected"]
            rel = (torch.sqrt(torch.mean((actual - expected) ** 2)) / expected.std()).item()
            print(
                f"\nLBF_PRECISION {request.node.callspec.id} pcc={compute_pcc(actual, expected):.6f} rel_rms={rel:.5f}"
            )
