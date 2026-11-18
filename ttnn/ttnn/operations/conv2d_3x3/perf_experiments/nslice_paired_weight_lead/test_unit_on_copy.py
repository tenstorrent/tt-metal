# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runs the op's own unit-test files against the private copy (op_copy/) with the experiment knobs set from the
environment, so a candidate walk is correctness-checked on every shape the op's suite covers (short slices, columns
that own no output, bm 1, multi-M-block, COL_MAJOR offset grids, the program-cache hit), not only the perf shapes.

    LO_UNIT=block LO_WALK=tall scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/nslice_paired_weight_lead/test_unit_on_copy.py \\
        --import-mode=prepend

LO_UNIT: block (test_conv2d_3x3_block_sharded.py) | layouts | height | main. LO_WALK / LO_LEAD set NSLICE_WALK /
NSLICE_LEAD in the copy. The real op package's public names and the two modules the tests reach through importlib are
redirected to the copy before the test module is imported.
"""

import importlib
import os
import sys

import pytest
import ttnn
import ttnn.operations.conv2d_3x3 as _real_pkg

# LO_PKG=op_patch: graduation.patch's tree (no knobs) instead of op_copy.
_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_paired_weight_lead." + os.environ.get("LO_PKG", "op_copy")
_copy_op = importlib.import_module(_PKG + ".conv2d_3x3")
_copy_pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
if hasattr(_copy_pd, "NSLICE_WALK"):
    _copy_pd.NSLICE_WALK = os.environ.get("LO_WALK", "slice")
    _copy_pd.NSLICE_LEAD = os.environ.get("LO_LEAD", "0") == "1"
    _copy_pd.select_blocking.cache_clear()

sys.modules["ttnn.operations.conv2d_3x3.conv2d_3x3"] = _copy_op
sys.modules["ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor"] = _copy_pd
for _name in _real_pkg.__all__:
    setattr(_real_pkg, _name, getattr(_copy_op, _name))

_UNIT = {
    "block": "test_conv2d_3x3_block_sharded",
    "layouts": "test_conv2d_3x3_layouts",
    "height": "test_conv2d_3x3_height_sharded",
    "main": "test_conv2d_3x3",
}[os.environ.get("LO_UNIT", "block")]
_mod = importlib.import_module(f"tests.ttnn.unit_tests.operations.conv2d_3x3.{_UNIT}")
assert _mod.conv2d_3x3 is _copy_op.conv2d_3x3
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__") and k not in globals()})

pytestmark = pytest.mark.use_module_device
