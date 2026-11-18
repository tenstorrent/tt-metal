# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load one of this experiment's op copies as a package (no __init__.py: `import ttnn` must not walk them) and
retarget an existing conv2d_3x3 test module at it. Host-only helpers; no torch, no device.

    op_base/ -> package npl_base: conv2d_3x3 at 3ca4eb28f46 (the baseline; opt-in NPL_ABLATE / NPL_BASE_SB41 knobs only)
    op_copy/ -> package npl_op_copy: 3ca4eb28f46 + the kw-fold narrow-planes candidate (= graduation.patch)
    op_exp/  -> package npl_exp: experiment copy with NPL_SB (subblock / plane shapes) and NPL_ABLATE knobs

NPL_OP=base / exp selects op_base / op_exp for `retarget` (default: op_copy)."""

import importlib
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGS = {"copy": ("npl_op_copy", "op_copy"), "base": ("npl_base", "op_base"), "exp": ("npl_exp", "op_exp")}


def copy_modules(which=None):
    which = which or os.environ.get("NPL_OP", "copy")
    pkg_name, sub = PKGS[which]
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(HERE / sub)]
        sys.modules[pkg_name] = pkg
    return (
        importlib.import_module(f"{pkg_name}.conv2d_3x3"),
        importlib.import_module(f"{pkg_name}.conv2d_3x3_program_descriptor"),
    )


def retarget(test_module_name, namespace, which=None):
    """Import a conv2d_3x3 test module, point every name it took from the real op (and its _op_module / _pd handles)
    at the selected copy, and re-export its test_* functions into `namespace` (the importing test file's globals)."""
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
