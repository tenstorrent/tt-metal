# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load one of this experiment's op copies as a package (no __init__.py: `import ttnn` must not walk them) and
retarget an existing conv2d_3x3 test module at it. Host-only helpers; no torch, no device.

    op_base/ -> package kws_base: conv2d_3x3 at 4f2f03f6d20 (HEAD, the baseline)
    op_copy/ -> package kws_op_copy: HEAD + the kw-fold start-up experiment knobs
    op_grad/ -> package kws_grad: HEAD + the graduation candidate (graduation.patch), no knobs

KWS_OP=base selects op_base for `retarget` (default: op_copy); base_unz / copy_unz: the same with every zone stripped."""

import importlib
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGS = {
    "copy": ("kws_op_copy", "op_copy"),
    "base": ("kws_base", "op_base"),
    "copy_unz": ("kws_op_copy_unz", "op_copy_unz"),
    "base_unz": ("kws_base_unz", "op_base_unz"),
    "grad": ("kws_grad", "op_grad"),
    "grad_unz": ("kws_grad_unz", "op_grad_unz"),
    "base_lz": ("kws_base_lz", "op_base_lz"),
    "grad_lz": ("kws_grad_lz", "op_grad_lz"),
    "copy_lz": ("kws_op_copy_lz", "op_copy_lz"),
}


def copy_modules(which=None):
    which = which or os.environ.get("KWS_OP", "copy")
    pkg_name, sub = PKGS[which]
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(HERE / sub)]
        sys.modules[pkg_name] = pkg
    op_mod = importlib.import_module(f"{pkg_name}.conv2d_3x3")
    pd_mod = importlib.import_module(f"{pkg_name}.conv2d_3x3_program_descriptor")
    # KWS_KNOBS="NAME=python-expr;...": descriptor knobs for a whole session (debug / bisect).
    for item in [k for k in os.environ.get("KWS_KNOBS", "").split(";") if k]:
        name, _, val = item.partition("=")
        assert hasattr(pd_mod, name), name
        setattr(pd_mod, name, eval(val))
    return op_mod, pd_mod


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
