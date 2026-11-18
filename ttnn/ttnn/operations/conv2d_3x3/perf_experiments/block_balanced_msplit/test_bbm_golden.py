# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""eval/golden_tests/conv2d_3x3/test_golden.py, unchanged, retargeted at this experiment's graduation copy (op_grad/):
the golden helpers and the axis-tagging wrapper call the copy's conv2d_3x3 / prepare_* (both dispatches).

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend -k BLOCK \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/block_balanced_msplit/test_bbm_golden.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bbm_copy import copy_modules, log_balanced  # noqa: E402

log_balanced(str(Path(__file__).resolve().parent / "runs" / "golden_balanced.log"))
_op, _ = copy_modules()
_real = importlib.import_module("ttnn.operations.conv2d_3x3")
_by_id = {id(getattr(_real, n)): getattr(_op, n) for n in _real.__all__}
for _name in (
    "eval.golden_tests.conv2d_3x3.axes",
    "eval.golden_tests.conv2d_3x3.helpers",
    "eval.golden_tests.conv2d_3x3.test_golden",
):
    _m = importlib.import_module(_name)
    for _attr, _value in list(vars(_m).items()):
        if id(_value) in _by_id and not _attr.startswith("__"):
            setattr(_m, _attr, _by_id[id(_value)])

from eval.golden_tests.conv2d_3x3.test_golden import test_op, test_op_loose  # noqa: E402,F401
