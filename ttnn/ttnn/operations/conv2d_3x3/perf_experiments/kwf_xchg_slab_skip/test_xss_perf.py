# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's perf harness (tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py), unchanged,
retargeted at op_copy/ (default) or op_base/ (XSS_OP=base; base_unz / copy_unz: zones stripped).

    XSS_OP=base_unz scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend \
        "ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_xchg_slab_skip/test_xss_perf.py::test_conv2d_3x3_perf_focus[res3_{1,2,3}_conv2]"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xss_copy import retarget  # noqa: E402

SWAPPED = retarget("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf", globals())
