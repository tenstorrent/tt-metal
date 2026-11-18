# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's perf harness (tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py), unchanged,
retargeted at this experiment's graduation candidate (op_grad/).

    scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend \\
        "ttnn/ttnn/operations/conv2d_3x3/perf_experiments/nslice_kernel_row_skip/test_nks_grad_perf.py::test_conv2d_3x3_perf_focus"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nks_grad import retarget  # noqa: E402

SWAPPED = retarget("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf", globals())
assert SWAPPED, "nothing retargeted"
