# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The op's HEIGHT-sharded and layouts unit tests (plus the interleaved file with SD_UNIT_ALL=1) re-bound to this dir's
op_copy/ (or another package with SD_UNIT_PKG), its knobs at their defaults (SLAB_DIRECT, SD_FUSED_BIAS on): every
face-direct shape compiles the slab-direct kernels. Run with --import-mode=prepend."""

import importlib
import os

_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.stem1_slab_tilize." + os.environ.get("SD_UNIT_PKG", "op_copy")
_op = importlib.import_module(_PKG + ".conv2d_3x3")
_pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
_UT = "tests.ttnn.unit_tests.operations.conv2d_3x3."

_MODS = ["test_conv2d_3x3_height_sharded", "test_conv2d_3x3_layouts"]
if os.environ.get("SD_UNIT_ALL"):
    _MODS += ["test_conv2d_3x3"]
for _mod_name in _MODS:
    _m = importlib.import_module(_UT + _mod_name)
    for _name in ("conv2d_3x3", "prepare_conv2d_bias", "prepare_conv2d_weights", "default_compute_kernel_config"):
        if hasattr(_m, _name):
            setattr(_m, _name, getattr(_op, _name))
    if hasattr(_m, "_op_module"):
        _m._op_module = _op
    if hasattr(_m, "_pd"):
        _m._pd = _pd
    for _name in dir(_m):
        if _name.startswith("test_"):
            globals()[f"test_{_mod_name[len('test_conv2d_3x3'):].lstrip('_') or 'main'}__{_name[5:]}"] = getattr(
                _m, _name
            )
