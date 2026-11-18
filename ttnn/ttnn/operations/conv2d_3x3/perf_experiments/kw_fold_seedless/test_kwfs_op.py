# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's perf harness (tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py), unchanged,
retargeted at this experiment's op copy (op_copy/: e40c435b625 + the seedless kw-fold candidate).

    scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend \
        "ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_seedless/test_kwfs_op.py::test_conv2d_3x3_perf_focus[stem_conv2]"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kwfs_copy import retarget  # noqa: E402

SWAPPED = retarget("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf", globals())
