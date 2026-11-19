# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load one of this experiment's op copies as a package (no __init__.py: `import ttnn` must not walk them) and
retarget an existing conv2d_3x3 test module at it. Host-only helpers; no torch, no device.

    op_base/ -> package xss_base: conv2d_3x3 at 039e18f6c81 (HEAD, the baseline)
    op_copy/ -> package xss_op_copy: HEAD + the kw-fold dead slab rows skipped (= graduation.patch)

XSS_OP=base selects op_base for `retarget` (default: op_copy); base_unz / copy_unz: the same with every zone stripped."""

import importlib
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGS = {
    "copy": ("xss_op_copy", "op_copy"),
    "base": ("xss_base", "op_base"),
    "copy_unz": ("xss_op_copy_unz", "op_copy_unz"),
    "base_unz": ("xss_base_unz", "op_base_unz"),
    "neg": ("xss_neg", "op_neg"),
    "basefix": ("xss_basefix", "op_basefix"),
}


def copy_modules(which=None):
    which = which or os.environ.get("XSS_OP", "copy")
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
