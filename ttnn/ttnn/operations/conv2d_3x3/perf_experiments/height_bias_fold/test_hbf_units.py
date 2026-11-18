# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runs the repo's conv2d_3x3 unit files (HBF_UNIT_FILES, default the HEIGHT and interleaved ones) against this
experiment's op copy with the variant HBF_VARIANT (default fold_utd): ttnn.operations.conv2d_3x3 and its descriptor
module are aliased to the copy before the unit modules are imported, so their tests call the copy.
  HBF_VARIANT=fold_utd scripts/run_safe_pytest.sh --run-all <this file> -p no:cacheprovider --import-mode=prepend
"""

import importlib
import os
import sys
from pathlib import Path

import ttnn  # noqa: F401  (the real package walk runs first; the aliases below replace it for the unit modules)

sys.path.insert(0, str(Path(__file__).parent))
import hbf_load  # noqa: E402

opc, pd, op_mod = hbf_load.load()
hbf_load.set_variant(pd, os.environ.get("HBF_VARIANT", "fold_utd"))
sys.modules["ttnn.operations.conv2d_3x3"] = opc
sys.modules["ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor"] = pd
sys.modules["ttnn.operations.conv2d_3x3.conv2d_3x3"] = op_mod

_UNITS = os.environ.get("HBF_UNIT_FILES", "test_conv2d_3x3_height_sharded,test_conv2d_3x3").split(",")
for _name in _UNITS:
    _mod = importlib.import_module(f"tests.ttnn.unit_tests.operations.conv2d_3x3.{_name}")
    for _k, _v in vars(_mod).items():
        if _k.startswith("test_") and callable(_v):
            globals()[f"{_k}__{_name.replace('test_conv2d_3x3', 'u') or 'u'}"] = _v


def test_hbf_units_use_the_copy():
    """The unit modules bound the copy's entry points and descriptor module, with the variant's knobs set."""
    for _name in _UNITS:
        mod = importlib.import_module(f"tests.ttnn.unit_tests.operations.conv2d_3x3.{_name}")
        assert mod.conv2d_3x3.__module__ == hbf_load.OPC + ".conv2d_3x3", mod.conv2d_3x3.__module__
    assert pd.KERNEL_DIR.parent.name == "op_copy" and pd.HBF_FOLD == {"fold_add": 2, "fold_add_c": 3}.get(
        os.environ.get("HBF_VARIANT", "fold_utd"), pd.HBF_FOLD
    )
