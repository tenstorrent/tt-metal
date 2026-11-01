# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""three_pass_restructure bake-off: LN3P_MODE=<0..3> [LN_ABLATE=read,write] scripts/run_safe_pytest.sh [--profile] <this file>."""

import pytest

import install3p  # noqa: F401  (patches the op before use)

from eval.golden_tests.layer_norm_tile.feature_spec import LOOSE_CASES
from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm

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
    "w_tail_8x512x800": _with(FOCUS, inputs=((8, 1, 512, 800),)),  # Wt=25: lane-group tails
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
