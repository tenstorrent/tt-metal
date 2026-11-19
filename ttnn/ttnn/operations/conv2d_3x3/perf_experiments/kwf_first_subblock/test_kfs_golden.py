# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The conv2d_3x3 golden suite (eval/golden_tests/conv2d_3x3/test_golden.py: test_op, test_op_loose), unchanged,
retargeted at op_copy/ (KFS_OP=base: op_base/): the helpers' and the observe wrapper's op handles are swapped.

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_first_subblock/test_kfs_golden.py -k HEIGHT
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kfs_copy import copy_modules  # noqa: E402

_op, _pd = copy_modules()
_helpers = importlib.import_module("eval.golden_tests.conv2d_3x3.helpers")
_axes = importlib.import_module("eval.golden_tests.conv2d_3x3.axes")
_real = _op.conv2d_3x3


def _counted(*a, **k):
    # Proof of retargeting: every dispatch through the copy is logged (runs/golden_calls.txt).
    with open(Path(__file__).resolve().parent / "runs" / "golden_calls.txt", "a") as f:
        f.write(f"{_op.__name__}\n")
    return _real(*a, **k)


_helpers._conv2d_3x3_raw = _counted
_helpers.prepare_conv2d_bias = _op.prepare_conv2d_bias
_helpers.prepare_conv2d_weights = _op.prepare_conv2d_weights
assert hasattr(_axes, "_raw")
_axes._raw = _counted

from eval.golden_tests.conv2d_3x3.test_golden import test_op, test_op_loose  # noqa: E402, F401
