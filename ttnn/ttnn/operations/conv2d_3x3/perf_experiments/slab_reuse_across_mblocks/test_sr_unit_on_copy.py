# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness: run a conv2d_3x3 unit-test file's tests against this experiment's op copy with the core-wide TILE slab
on (SR_VARIANT, default core).

The unit file's source is loaded with its op imports pointed at the copy and executed into this module, so pytest
collects its tests unchanged. SR_UNIT picks the file (height | block | base | layouts). Several unit tests pin a
multi-M-block blocking by calling `_footprint` directly (one-tile-row M-blocks); with SR_FORCE=1 (default) such a
call takes the core slab wherever it has fewer rows than the per-block slabs, so those pinned multi-block tests run
the core slab too (L1 is not re-checked there, as in the unit tests themselves).

  SR_UNIT=height scripts/run_safe_pytest.sh --run-all \\
      ttnn/ttnn/operations/conv2d_3x3/perf_experiments/slab_reuse_across_mblocks/test_sr_unit_on_copy.py \\
      --import-mode=prepend
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[5]
sys.path.insert(0, str(HERE))
import sr_load  # noqa: E402

_opc, _copy_pd, _copy_op = sr_load.load()
OPC = sr_load.OPC
sr_load.set_variant(_copy_pd, os.environ.get("SR_VARIANT", "core"))

if os.environ.get("SR_FORCE", "1") == "1":
    _real_footprint = _copy_pd._footprint

    def _footprint_core_where_capable(g, core_m_tiles, bm, *args, core_slab=None, **kwargs):
        if core_slab is None:
            core_slab = _copy_pd.core_slab_capable(g, core_m_tiles, bm)
            if core_slab:
                print(f"SR_CORE_SLAB_FORCED core_m_tiles={core_m_tiles} bm={bm} k_cols={g.k_cols}")
        return _real_footprint(g, core_m_tiles, bm, *args, core_slab=core_slab, **kwargs)

    _copy_pd._footprint = _footprint_core_where_capable

_UNIT_FILES = {
    "height": "test_conv2d_3x3_height_sharded.py",
    "block": "test_conv2d_3x3_block_sharded.py",
    "base": "test_conv2d_3x3.py",
    "layouts": "test_conv2d_3x3_layouts.py",
}
_src_path = REPO / "tests/ttnn/unit_tests/operations/conv2d_3x3" / _UNIT_FILES[os.environ.get("SR_UNIT", "height")]
_src = _src_path.read_text()
for old, new in [
    ('"ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor"', f'"{OPC}.conv2d_3x3_program_descriptor"'),
    ('"ttnn.operations.conv2d_3x3.conv2d_3x3"', f'"{OPC}.conv2d_3x3"'),
    ("from ttnn.operations.conv2d_3x3 import", f"from {OPC}.conv2d_3x3 import"),
]:
    _src = _src.replace(old, new)
assert "ttnn.operations.conv2d_3x3" not in _src, "an op import still points at the real op"
exec(compile(_src, str(_src_path), "exec"), globals())
