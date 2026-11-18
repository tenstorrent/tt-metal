# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3.py, unchanged, retargeted at this experiment's graduation copy
(op_grad/: 3ca4eb28f46 + the work-balanced BLOCK M split, knob-free).

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/block_balanced_msplit/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bbm_copy import log_balanced, retarget  # noqa: E402

log_balanced(str(Path(__file__).resolve().parent / "runs" / "unit_balanced.log"))

SWAPPED = retarget("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3", globals())
