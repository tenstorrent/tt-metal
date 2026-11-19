# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load this experiment's op copy as package `hwr_op_copy` (no __init__.py: `import ttnn` must not walk it), and
retarget an existing conv2d_3x3 test module at it. HWR_COPY picks the directory: unset = op_copy/ (the candidate,
zoned), "nz" = op_copy_nz/ (op_copy with every MaybeDeviceZoneScope line stripped), "base" = op_head/ (the op at HEAD
6981db2a54c, zoned), "base_nz" = op_head_nz/ (HEAD unzoned), "grad" = grad/ (the graduation candidate: HEAD +
graduation.patch, no HWR_* switches), "grad_nz" = grad_nz/; make_nz.sh builds the *_nz copies. Host-only helpers."""

import importlib
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = "hwr_op_copy"


def copy_dir():
    v = os.environ.get("HWR_COPY", "")
    return (
        HERE
        / {
            "": "op_copy",
            "nz": "op_copy_nz",
            "base": "op_head",
            "base_nz": "op_head_nz",
            "grad": "grad",
            "grad_nz": "grad_nz",
        }[v]
    )


def copy_modules():
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(copy_dir())]
        sys.modules[PKG] = pkg
    return importlib.import_module(f"{PKG}.conv2d_3x3"), importlib.import_module(f"{PKG}.conv2d_3x3_program_descriptor")


def retarget(test_module_name, namespace):
    """Import a conv2d_3x3 test module, point every name it took from the real op at the copy, and re-export its
    test_* functions into `namespace` (the importing test file's globals)."""
    op_mod, pd_mod = copy_modules()
    real_pkg = importlib.import_module("ttnn.operations.conv2d_3x3")
    real_op = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")
    real_pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")
    m = importlib.import_module(test_module_name)
    swapped = []
    for name, value in list(vars(m).items()):
        if value is real_op:
            setattr(m, name, op_mod)
        elif value is real_pd:
            setattr(m, name, pd_mod)
        elif name in getattr(real_pkg, "__all__", ()) and value is getattr(real_pkg, name, None):
            setattr(m, name, getattr(op_mod, name))
        else:
            continue
        swapped.append(name)
    for name, value in vars(m).items():
        if (name.startswith("test_") and callable(value)) or name.startswith("pytestmark"):
            namespace[name] = value
    return swapped
