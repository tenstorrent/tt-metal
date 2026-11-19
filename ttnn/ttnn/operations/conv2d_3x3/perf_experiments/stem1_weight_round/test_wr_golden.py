# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""eval/golden_tests/conv2d_3x3/test_golden.py, unchanged, retargeted at this experiment's op copy (op_grad/, or
WR_UNIT_PKG): the golden helpers and the axis-tagging wrapper call the copy's conv2d_3x3 / prepare_* (both dispatches).

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend -k "..." \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/stem1_weight_round/test_wr_golden.py
"""

import importlib
import os

_op = importlib.import_module(
    "ttnn.operations.conv2d_3x3.perf_experiments.stem1_weight_round." + os.environ.get("WR_UNIT_PKG", "op_grad")
)
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
