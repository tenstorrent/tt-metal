# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Load this experiment's op copies as packages (no __init__.py: `import ttnn` must not walk them). Host-only.

    op_base/   -> tsrg_base: conv2d_3x3 at HEAD 039e18f6c81 (end of Perf 10)
    grad_copy/ -> tsrg_grad: op_base + graduation.patch"""

import importlib
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PKGS = {"grad": ("tsrg_grad", "grad_copy"), "base": ("tsrg_base", "op_base")}


def copy_modules(which):
    pkg_name, sub = PKGS[which]
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(HERE / sub)]
        sys.modules[pkg_name] = pkg
    return (
        importlib.import_module(f"{pkg_name}.conv2d_3x3"),
        importlib.import_module(f"{pkg_name}.conv2d_3x3_program_descriptor"),
    )


def retarget(test_module_name, namespace, which="grad"):
    """Import a conv2d_3x3 test module, point every name it took from the real op at the copy, and re-export its
    test_* functions into `namespace`."""
    op_mod, pd_mod = copy_modules(which)
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
