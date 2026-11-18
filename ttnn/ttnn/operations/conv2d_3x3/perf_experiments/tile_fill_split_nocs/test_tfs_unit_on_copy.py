# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness: run a conv2d_3x3 unit-test file's tests against this experiment's op copy, split TILE fill on
(TFS_VARIANT, default split_poll_rx).

The unit file's source is loaded with its op imports pointed at the copy and executed into this module, so pytest
collects its tests unchanged. TFS_UNIT picks the file (height | block | base | layouts).

  TFS_UNIT=height scripts/run_safe_pytest.sh --run-all \\
      ttnn/ttnn/operations/conv2d_3x3/perf_experiments/tile_fill_split_nocs/test_tfs_unit_on_copy.py --import-mode=prepend
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[5]
sys.path.insert(0, str(HERE))
import tfs_load  # noqa: E402

_opc, _copy_pd, _copy_op = tfs_load.load()
OPC = tfs_load.OPC
# TFS_VARIANT: the knob set of test_tile_fill_split.VARIANTS (default: the recommended split_poll_rx).
_KNOBS = {
    "base": {},
    "split": dict(TILE_FILL_SPLIT_NOCS=True),
    "split_poll": dict(TILE_FILL_SPLIT_NOCS=True, TILE_FILL_POLL_RECEIVE=True),
    "split_poll_rx": dict(TILE_FILL_SPLIT_NOCS=True, TILE_FILL_POLL_RECEIVE=True, TILE_FILL_RECEIVERS_ONLY=True),
}[os.environ.get("TFS_VARIANT", "split_poll_rx")]
for _k, _v in _KNOBS.items():
    setattr(_copy_pd, _k, _v)

_UNIT_FILES = {
    "height": "test_conv2d_3x3_height_sharded.py",
    "block": "test_conv2d_3x3_block_sharded.py",
    "base": "test_conv2d_3x3.py",
    "layouts": "test_conv2d_3x3_layouts.py",
}
_src_path = REPO / "tests/ttnn/unit_tests/operations/conv2d_3x3" / _UNIT_FILES[os.environ.get("TFS_UNIT", "height")]
_src = _src_path.read_text()
for old, new in [
    ('"ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor"', f'"{OPC}.conv2d_3x3_program_descriptor"'),
    ('"ttnn.operations.conv2d_3x3.conv2d_3x3"', f'"{OPC}.conv2d_3x3"'),
    ("from ttnn.operations.conv2d_3x3 import", f"from {OPC}.conv2d_3x3 import"),
]:
    _src = _src.replace(old, new)
assert "ttnn.operations.conv2d_3x3" not in _src, "an op import still points at the real op"
exec(compile(_src, str(_src_path), "exec"), globals())
