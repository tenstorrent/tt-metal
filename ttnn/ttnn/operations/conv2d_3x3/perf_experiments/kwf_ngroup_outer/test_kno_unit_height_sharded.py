# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_height_sharded.py, unchanged, retargeted at this experiment's op copy.

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_ngroup_outer/test_kno_unit_height_sharded.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kno_copy import retarget  # noqa: E402

# With KNO_OP_DIR=op_grad, the graduated test file (graduation_tests.patch applied).
_MOD = (
    str(Path(__file__).resolve().parent / "tests_grad" / "test_conv2d_3x3_height_sharded.py.in")
    if os.environ.get("KNO_OP_DIR") == "op_grad"
    else "tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_height_sharded"
)
SWAPPED = retarget(_MOD, globals())
