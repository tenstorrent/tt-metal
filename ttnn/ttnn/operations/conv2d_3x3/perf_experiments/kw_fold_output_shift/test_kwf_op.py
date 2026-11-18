# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's perf harness (tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py), unchanged,
retargeted at this experiment's op copy (op_copy/: 5f23b256d1a + the round-6 ablation hooks, env-gated).

    CONV2D_3X3_PERF_ABLATION=1 CONV2D_3X3_ABLATE=fill,weights scripts/run_safe_pytest.sh --profile --run-all \
        --import-mode=prepend \
        "ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_output_shift/test_kwf_op.py::test_conv2d_3x3_perf_focus[res3_1]"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kwf_copy import retarget  # noqa: E402

SWAPPED = retarget("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf", globals())
