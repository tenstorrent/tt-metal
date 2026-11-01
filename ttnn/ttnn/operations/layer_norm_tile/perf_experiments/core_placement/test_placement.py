# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""core_placement bake-off: LN_PLACEMENT=<variant> scripts/run_safe_pytest.sh [--profile] <this file>.

Perf rows = the 8 LOOSE_CASES perf rows; extra rows cover fp32_dest_acc_en=False, h_non_aligned,
rows < cores and ragged splits. Correctness vs torch at the golden tolerance for every row.
"""

import pytest
import ttnn

from eval.golden_tests.layer_norm_tile.feature_spec import LOOSE_CASES
from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm

from placement import install

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


def _with(case, **kw):
    out = dict(case)
    out["extras"] = dict(case["extras"])
    for k, v in kw.items():
        if k == "inputs":
            out[k] = v
        elif k in ("residual_buffer_type", "output_buffer_type"):
            out["extras"][k] = v
        else:
            out[k] = v
    return out


FOCUS = PERF_CASES[0]
EXTRA_CASES = {
    "focus1_fp32dest_off": _with(FOCUS, fp32_dest_acc_en=False),
    "h_non_aligned_3x500": _with(FOCUS, inputs=((3, 1, 500, 768),)),  # 48 rows < 110 cores
    "ragged_8x520": _with(FOCUS, inputs=((8, 1, 520, 768),)),  # 136 rows: 26 big cores
    "ragged_2x1800": _with(FOCUS, inputs=((2, 1, 1800, 768),)),  # 114 rows: 4 big cores
    "many_rows_8x1024_dram": _with(PERF_CASES[1], inputs=((8, 1, 1024, 768),)),  # 256 rows: 2-3/core
    "tiny_1x64": _with(FOCUS, inputs=((1, 1, 64, 768),)),  # 2 rows
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
