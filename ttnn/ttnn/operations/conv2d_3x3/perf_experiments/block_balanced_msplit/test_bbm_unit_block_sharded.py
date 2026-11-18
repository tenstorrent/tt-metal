# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_block_sharded.py, retargeted at this experiment's
graduation copy (op_grad/: 3ca4eb28f46 + the work-balanced BLOCK M split, knob-free). One test changes, as in
unit_tests.patch: column_weight_mcast_ragged_row pins the uniform split, whose ragged last row is its premise.

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/block_balanced_msplit/test_bbm_unit_block_sharded.py
"""

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bbm_copy import copy_modules, log_balanced, retarget  # noqa: E402

log_balanced(str(Path(__file__).resolve().parent / "runs" / "unit_balanced.log"))

SWAPPED = retarget("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_block_sharded", globals())
_ragged_row = globals()["test_conv2d_3x3_block_sharded_column_weight_mcast_ragged_row"]


@functools.wraps(_ragged_row)
def test_conv2d_3x3_block_sharded_column_weight_mcast_ragged_row(dtype, layout, device, monkeypatch):
    op_mod, _ = copy_modules()
    monkeypatch.setattr(op_mod, "balanced_block_ranges", lambda *args: None)
    _ragged_row(dtype, layout, device, monkeypatch)
