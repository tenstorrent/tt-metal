# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness: run a conv2d_3x3 unit-test file's tests against this experiment's op copy, deep ring on.

The unit file's source is loaded with its op imports pointed at the copy and executed into this module, so pytest
collects its tests unchanged. DIR_UNIT picks the file (height | block | base | layouts); DIR_DEEP the ring target
("bm" default; see op_copy IM2COL_RM_DEEP). Every select_blocking result is appended to runs/unit_<ts>.jsonl, so the
cases that actually ran a deeper ring are visible.

  DIR_UNIT=height scripts/run_safe_pytest.sh --run-all \\
      ttnn/ttnn/operations/conv2d_3x3/perf_experiments/deep_im2col_ring/test_unit_on_copy.py --import-mode=prepend
"""

import importlib
import json
import os
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[5]
OPC = "deep_im2col_ring_op_copy"
if OPC not in sys.modules:
    _pkg = types.ModuleType(OPC)
    _pkg.__path__ = [str(HERE / "op_copy")]
    sys.modules[OPC] = _pkg
_copy_pd = importlib.import_module(OPC + ".conv2d_3x3_program_descriptor")
_copy_op = importlib.import_module(OPC + ".conv2d_3x3")
_deep = os.environ.get("DIR_DEEP", "bm")
_copy_pd.IM2COL_RM_DEEP = int(_deep) if _deep.isdigit() else (None if _deep == "none" else _deep)

_UNIT_LOG = HERE / "runs" / f"unit_{int(time.time())}_{os.getpid()}.jsonl"
_real_select = _copy_op.select_blocking


def _recording_select(g, grid, l1_budget):
    b = _real_select(g, grid, l1_budget)
    _UNIT_LOG.parent.mkdir(exist_ok=True)
    with open(_UNIT_LOG, "a") as f:
        f.write(
            json.dumps(
                dict(
                    geom=[g.input_height, g.input_width, g.c_in_p, g.c_out, g.stride, g.dilation, int(g.tile_input)],
                    k_cols=g.k_cols,
                    bm=b.block_m_tiles,
                    bk=b.block_k_tiles,
                    nk=b.num_k_blocks,
                    depth=b.im2col_rm_depth_rows,
                    nslice=b.nslice,
                    slack=l1_budget - b.footprint_bytes,
                )
            )
            + "\n"
        )
    return b


_copy_op.select_blocking = _recording_select

_UNIT_FILES = {
    "height": "test_conv2d_3x3_height_sharded.py",
    "block": "test_conv2d_3x3_block_sharded.py",
    "base": "test_conv2d_3x3.py",
    "layouts": "test_conv2d_3x3_layouts.py",
}
_src_path = REPO / "tests/ttnn/unit_tests/operations/conv2d_3x3" / _UNIT_FILES[os.environ.get("DIR_UNIT", "height")]
_src = _src_path.read_text()
for old, new in [
    ('"ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor"', f'"{OPC}.conv2d_3x3_program_descriptor"'),
    ('"ttnn.operations.conv2d_3x3.conv2d_3x3"', f'"{OPC}.conv2d_3x3"'),
    ("from ttnn.operations.conv2d_3x3 import", f"from {OPC}.conv2d_3x3 import"),
]:
    _src = _src.replace(old, new)
assert "ttnn.operations.conv2d_3x3" not in _src, "an op import still points at the real op"
exec(compile(_src, str(_src_path), "exec"), globals())
