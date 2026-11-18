# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load this experiment's graduation candidate (op_grad/: conv2d_3x3 at e40c435b625 + graduation.patch) as package
`nks_op_grad` (no __init__.py: `import ttnn` must not walk it), and retarget existing conv2d_3x3 test / golden modules at
it. Host-only helpers; no torch, no device."""

import importlib
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = "nks_op_grad"


def copy_modules():
    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [str(HERE / "op_grad")]
        sys.modules[PKG] = pkg
    op_mod = importlib.import_module(f"{PKG}.conv2d_3x3")
    pd_mod = importlib.import_module(f"{PKG}.conv2d_3x3_program_descriptor")
    assert Path(pd_mod.KERNEL_DIR).parent == HERE / "op_grad", pd_mod.KERNEL_DIR
    return op_mod, pd_mod


def retarget(module_name, namespace=None):
    """Import a module, point every object it took from the real op (by identity: the op / descriptor modules and every
    exported name, under any alias) at the copy, and re-export its test_* functions into `namespace`."""
    op_mod, pd_mod = copy_modules()
    real_pkg = importlib.import_module("ttnn.operations.conv2d_3x3")
    real_op = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")
    real_pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")
    swap = {id(real_op): op_mod, id(real_pd): pd_mod}
    for name in getattr(real_pkg, "__all__", ()):
        swap[id(getattr(real_pkg, name))] = getattr(op_mod, name)
    m = importlib.import_module(module_name)
    swapped = []
    for name, value in list(vars(m).items()):
        if id(value) in swap and not isinstance(value, (int, str, bool, type(None))):
            setattr(m, name, swap[id(value)])
            swapped.append(name)
    if namespace is not None:
        for name, value in vars(m).items():
            if name.startswith("test_") and callable(value):
                namespace[name] = value
    return swapped
