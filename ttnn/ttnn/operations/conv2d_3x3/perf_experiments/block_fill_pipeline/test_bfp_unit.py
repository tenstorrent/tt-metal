# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness: run a conv2d_3x3 unit-test file's tests against this experiment's op copy with a variant's knobs
(BFP_UNIT_VARIANT, default `rec`; the knob sets of test_bfp.VARIANTS).

The unit file's source is loaded with its op imports pointed at the copy and executed into this module, so pytest
collects its tests unchanged. BFP_UNIT picks the file (height | block | base | layouts).

  BFP_UNIT=block scripts/run_safe_pytest.sh --run-all \\
      ttnn/ttnn/operations/conv2d_3x3/perf_experiments/block_fill_pipeline/test_bfp_unit.py --import-mode=prepend
"""

import importlib
import os
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[5]
_COPY = os.environ.get("BFP_PKG", "op_copy")
PKG = "ttnn.operations.conv2d_3x3.perf_experiments.block_fill_pipeline." + _COPY
_pd = importlib.import_module(PKG + ".conv2d_3x3_program_descriptor")

_VARIANTS_SRC = (HERE / "test_bfp.py").read_text()
_ns = {}
# Only the knob tables (no ttnn / device code): NZL, KNOB_DEFAULTS and VARIANTS.
exec(
    compile(
        _VARIANTS_SRC[_VARIANTS_SRC.index("KNOB_DEFAULTS = dict(") : _VARIANTS_SRC.index("def apply_variant")],
        "test_bfp.py",
        "exec",
    ),
    _ns,
)
_variant = os.environ.get("BFP_UNIT_VARIANT", "rec")
if _COPY != "grad":
    for _k, _v in {**_ns["KNOB_DEFAULTS"], **_ns["VARIANTS"][_variant]}.items():
        setattr(_pd, _k, _v)
_pd.select_blocking.cache_clear()

_UNIT_FILES = {
    "height": "test_conv2d_3x3_height_sharded.py",
    "block": "test_conv2d_3x3_block_sharded.py",
    "base": "test_conv2d_3x3.py",
    "layouts": "test_conv2d_3x3_layouts.py",
}
_src_path = REPO / "tests/ttnn/unit_tests/operations/conv2d_3x3" / _UNIT_FILES[os.environ.get("BFP_UNIT", "block")]
_src = _src_path.read_text()
_n = 0
for old, new in [
    ('"ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor"', f'"{PKG}.conv2d_3x3_program_descriptor"'),
    ('"ttnn.operations.conv2d_3x3.conv2d_3x3"', f'"{PKG}.conv2d_3x3"'),
    ("from ttnn.operations.conv2d_3x3 import", f"from {PKG}.conv2d_3x3 import"),
]:
    _n += _src.count(old)
    _src = _src.replace(old, new)
assert _n > 0
assert "ttnn.operations.conv2d_3x3 import" not in _src.replace(f"{PKG}.conv2d_3x3 import", "")
assert '"ttnn.operations.conv2d_3x3.conv2d_3x3' not in _src, "an op import still points at the real op"
exec(compile(_src, str(_src_path), "exec"), globals())
