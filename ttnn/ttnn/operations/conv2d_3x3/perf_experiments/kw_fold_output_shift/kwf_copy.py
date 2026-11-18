# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load this experiment's op copy (op_copy/: conv2d_3x3 at 5f23b256d1a + the round-6 ablation hooks (+ the kw-fold path once built)) as package
`kwf_op_copy` (no __init__.py: `import ttnn` must not walk it), and retarget an existing conv2d_3x3 test module at it.
Host-only helpers; no torch, no device."""

import importlib
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = "kwf_op_copy"


def copy_modules():
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(HERE / "op_copy")]
        sys.modules[PKG] = pkg
    return importlib.import_module(f"{PKG}.conv2d_3x3"), importlib.import_module(f"{PKG}.conv2d_3x3_program_descriptor")


def retarget(test_module_name, namespace):
    """Import a conv2d_3x3 test module, point every name it took from the real op (and its _op_module / _pd handles)
    at the copy, and re-export its test_* functions into `namespace` (the importing test file's globals)."""
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
        if name.startswith("test_") and callable(value):
            namespace[name] = value
    return swapped
