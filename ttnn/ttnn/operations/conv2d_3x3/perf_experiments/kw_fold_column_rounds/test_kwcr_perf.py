# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's perf focus test (tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py,
test_conv2d_3x3_perf_focus), unchanged, retargeted at one op copy and filtered by env (no -k: the profiler wrapper
re-joins the command line, and the stock ids carry braces).

    KWCR_COPY=op_cand KWCR_CASES=inst_r2_fc1,center1 scripts/run_safe_pytest.sh --profile --run-all \
        --import-mode=prepend ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_column_rounds/test_kwcr_perf.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kwcr_copy import focus_cases, retarget  # noqa: E402

_NS = {}
SWAPPED = retarget(
    "tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf", _NS, os.environ.get("KWCR_COPY", "op_cand")
)
_CASES, _IDS = focus_cases(os.environ.get("KWCR_CASES", ""))


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_kwcr_focus(case, device):
    _NS["test_conv2d_3x3_perf_focus"](case, device)
