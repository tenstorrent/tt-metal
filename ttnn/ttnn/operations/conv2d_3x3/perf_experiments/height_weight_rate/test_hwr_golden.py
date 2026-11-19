# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""eval/golden_tests/conv2d_3x3/test_golden.py and test_regression.py, unchanged, retargeted at this experiment's op
copy (HWR_COPY, hwr_copy.py): the helpers' prepare_conv2d_* and raw op, and axes.py's observed wrapper target, are
pointed at the copy before the tests run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hwr_copy import copy_modules  # noqa: E402

_op, _ = copy_modules()

import eval.golden_tests.conv2d_3x3.axes as _axes  # noqa: E402
import eval.golden_tests.conv2d_3x3.helpers as _helpers  # noqa: E402

_axes._raw = _op.conv2d_3x3
_helpers._conv2d_3x3_raw = _op.conv2d_3x3
_helpers.prepare_conv2d_weights = _op.prepare_conv2d_weights
_helpers.prepare_conv2d_bias = _op.prepare_conv2d_bias

import eval.golden_tests.conv2d_3x3.test_regression as _reg  # noqa: E402

_reg.conv2d_3x3 = _op.conv2d_3x3
_reg.prepare_conv2d_weights = _op.prepare_conv2d_weights
_reg.prepare_conv2d_bias = _op.prepare_conv2d_bias

from eval.golden_tests.conv2d_3x3.test_golden import *  # noqa: E402,F401,F403
from eval.golden_tests.conv2d_3x3.test_regression import *  # noqa: E402,F401,F403
