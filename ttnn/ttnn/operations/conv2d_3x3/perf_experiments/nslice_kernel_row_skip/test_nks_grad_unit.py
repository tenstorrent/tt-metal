# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's unit tests, unchanged, retargeted at this experiment's graduation candidate (op_grad/).
NKS_UNIT selects the module (default test_conv2d_3x3_block_sharded).

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/nslice_kernel_row_skip/test_nks_grad_unit.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nks_grad import retarget  # noqa: E402

SWAPPED = retarget(
    "tests.ttnn.unit_tests.operations.conv2d_3x3." + os.environ.get("NKS_UNIT", "test_conv2d_3x3_block_sharded"),
    globals(),
)
assert SWAPPED, "nothing retargeted"
