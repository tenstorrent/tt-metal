# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""affine_mcast bake-off: LN_AFFINE=<variant> scripts/run_safe_pytest.sh [--profile] <this file>.

Perf rows = the 8 LOOSE_CASES perf rows; extra rows sweep affine layout x affine placement on the
focus shape, fp32_dest_acc_en=False, h_non_aligned (48 active cores, 2 rectangles) and a 1-core call.
Correctness vs torch at the golden tolerance for every row.
"""

import sys
from pathlib import Path

import pytest
import ttnn

from eval.golden_tests.layer_norm_tile.feature_spec import LOOSE_CASES
from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm

sys.path.insert(0, str(Path(__file__).parent))
from affine_mcast import install  # noqa: E402

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
    out.update(kw)
    return out


FOCUS = PERF_CASES[0]
DRAM_RES = PERF_CASES[1]
EXTRA_CASES = {
    "focus1_affineRM_DRAM": _with(FOCUS, affine_buffer_type=ttnn.BufferType.DRAM),
    "focus1_affineTILE_L1": _with(FOCUS, affine_layout=ttnn.TILE_LAYOUT),
    "focus1_affineTILE_DRAM": _with(FOCUS, affine_layout=ttnn.TILE_LAYOUT, affine_buffer_type=ttnn.BufferType.DRAM),
    "focus2_affineRM_DRAM": _with(DRAM_RES, affine_buffer_type=ttnn.BufferType.DRAM),
    "focus2_affineTILE_L1": _with(DRAM_RES, affine_layout=ttnn.TILE_LAYOUT),
    "focus1_fp32dest_off": _with(FOCUS, fp32_dest_acc_en=False),
    "h_non_aligned_3x500": _with(FOCUS, inputs=((3, 1, 500, 768),)),
    "h_non_aligned_3x500_affineTILE_DRAM": _with(
        FOCUS, inputs=((3, 1, 500, 768),), affine_layout=ttnn.TILE_LAYOUT, affine_buffer_type=ttnn.BufferType.DRAM
    ),
    "one_core_1x32": _with(FOCUS, inputs=((1, 1, 32, 768),)),
    "two_core_1x64": _with(FOCUS, inputs=((1, 1, 64, 768),)),
    "many_rows_8x1024_dram": _with(DRAM_RES, inputs=((8, 1, 1024, 768),)),
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
