# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's unit tests, unchanged, retargeted at op_grad/ (KWS_OP=base: op_base/, copy: op_copy/).

    KWS_OP=grad scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_startup/test_kws_unit.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kws_copy import retarget  # noqa: E402

_BASE = "tests.ttnn.unit_tests.operations.conv2d_3x3."
_WHICH = os.environ.get("KWS_OP", "grad")
_MODULES = [
    m
    for m in os.environ.get(
        "KWS_UNIT_MODULES",
        "test_conv2d_3x3_kw_fold,test_conv2d_3x3_height_sharded,test_conv2d_3x3_layouts,test_conv2d_3x3",
    ).split(",")
    if m
]
SWAPPED = {name: retarget(_BASE + name, globals(), _WHICH) for name in _MODULES}
