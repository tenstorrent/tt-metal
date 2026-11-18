# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The op's whole unit-test dir (interleaved, HEIGHT, layouts, BLOCK) re-bound to op_grad/ (the real op at 5f23b256d1a +
graduation.patch): every shape compiles the patched reader. Run with --import-mode=prepend."""

import importlib

_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_center_taps.op_grad"
_op = importlib.import_module(_PKG + ".conv2d_3x3")
_pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
_UT = "tests.ttnn.unit_tests.operations.conv2d_3x3."

for _mod_name in (
    "test_conv2d_3x3",
    "test_conv2d_3x3_height_sharded",
    "test_conv2d_3x3_layouts",
    "test_conv2d_3x3_block_sharded",
):
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
