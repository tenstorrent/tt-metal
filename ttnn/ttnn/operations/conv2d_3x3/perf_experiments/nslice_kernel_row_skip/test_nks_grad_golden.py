# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's golden suite (eval/golden_tests/conv2d_3x3/test_golden.py), unchanged, with its op calls (axes.py's
observed wrapper and helpers.py's raw / prepare calls) retargeted at this experiment's graduation candidate (op_grad/).

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/nslice_kernel_row_skip/test_nks_grad_golden.py -k BLOCK
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nks_grad import retarget  # noqa: E402

AXES = retarget("eval.golden_tests.conv2d_3x3.axes")
HELPERS = retarget("eval.golden_tests.conv2d_3x3.helpers")
assert "_raw" in AXES and "_conv2d_3x3_raw" in HELPERS, (AXES, HELPERS)
SWAPPED = retarget("eval.golden_tests.conv2d_3x3.test_golden", globals())
