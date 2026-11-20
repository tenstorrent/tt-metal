# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's unit tests, unchanged, retargeted at one of this experiment's copies (SWS_OP, default copy).
SWS_RESIDENT=1 turns the copy's S2W_RESIDENT knob on for the whole session; SWS_LOG=<file> logs every server-fed
non-kw-fold walk the descriptor builds (op_copy only).

    SWS_OP=copy SWS_RESIDENT=1 scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/s2_weight_slots/test_sws_unit.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sws_copy import copy_modules, retarget  # noqa: E402

_BASE = "tests.ttnn.unit_tests.operations.conv2d_3x3."
_WHICH = os.environ.get("SWS_OP", "copy")
_MODULES = [
    m
    for m in os.environ.get(
        "SWS_UNIT_MODULES",
        "test_conv2d_3x3_stride2_taps,test_conv2d_3x3_height_sharded,test_conv2d_3x3_kw_fold,test_conv2d_3x3_layouts",
    ).split(",")
    if m
]
if os.environ.get("SWS_RESIDENT") == "1":
    copy_modules(_WHICH)[1].S2W_RESIDENT = True
SWAPPED = {name: retarget(_BASE + name, globals(), _WHICH) for name in _MODULES}
