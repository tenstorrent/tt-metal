# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's perf harness (tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py), unchanged,
retargeted at this experiment's op copy (op_copy/: 3ca4eb28f46 + the kw-fold halo exchange candidate).

    scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend \
        "ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_halo_exchange/test_khx_op.py::test_conv2d_3x3_perf_focus[res3_1_conv2]"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from khx_copy import retarget  # noqa: E402

SWAPPED = retarget("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf", globals())
