# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The BLOCK unit tests (tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_block_sharded.py) on this dir's
op copy with the recommended knobs (FUO_KNOBS, default k4w8ww4: K-split untilize batch <= 4 tile-rows, K-split reader
fill window 8). Correctness gate for the candidate; no profiling."""

import importlib
import os

import tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_block_sharded as unit

_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.block_fill_untilize_overlap.op_copy"
_op = importlib.import_module(_PKG + ".conv2d_3x3")
_pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")

if os.environ.get("FUO_KNOBS", "k4w8ww4") == "k4w8ww4":
    _pd.FUO_UB = 4
    _pd.FUO_KSPLIT_WINDOW = 8
    _pd.FUO_KSPLIT_WRITER_WINDOW = 4

unit.conv2d_3x3 = _op.conv2d_3x3
unit.prepare_conv2d_weights = _op.prepare_conv2d_weights
unit.prepare_conv2d_bias = _op.prepare_conv2d_bias
unit._op_module = _op
unit._pd = _pd

from tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_block_sharded import *  # noqa: E402,F401,F403
