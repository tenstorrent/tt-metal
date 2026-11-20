# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load one of this experiment's op copies as a package (no __init__.py: `import ttnn` must not walk them) and
retarget an existing conv2d_3x3 test module at it. Host-only helpers; no torch, no device.

    op_base/ -> package knw_base: conv2d_3x3 at 1a04504f992 (HEAD, the baseline)
    op_copy/ -> package knw_op_copy: HEAD + the kw-fold narrow-walk experiment knobs
    op_grad/ -> package knw_grad: HEAD + the graduation candidate (graduation.patch), no knobs
    op_head/ -> package knw_head: HEAD exactly (no ablation hooks), the sweep's baseline

KNW_OP=base selects op_base for `retarget` (default: op_copy); base_unz / copy_unz: the same with every zone stripped."""

import importlib
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGS = {
    "copy": ("knw_op_copy", "op_copy"),
    "base": ("knw_base", "op_base"),
    "copy_unz": ("knw_op_copy_unz", "op_copy_unz"),
    "base_unz": ("knw_base_unz", "op_base_unz"),
    "grad": ("knw_grad", "op_grad"),
    "head": ("knw_head", "op_head"),
    "fsr1": ("knw_fsr1", "op_fsr1"),
    "fsr1_unz": ("knw_fsr1_unz", "op_fsr1_unz"),
    "head_unz": ("knw_head_unz", "op_head_unz"),
    "grad_unz": ("knw_grad_unz", "op_grad_unz"),
    "base_lz": ("knw_base_lz", "op_base_lz"),
    "grad_lz": ("knw_grad_lz", "op_grad_lz"),
    "copy_lz": ("knw_op_copy_lz", "op_copy_lz"),
}


def copy_modules(which=None):
    which = which or os.environ.get("KNW_OP", "copy")
    pkg_name, sub = PKGS[which]
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(HERE / sub)]
        sys.modules[pkg_name] = pkg
    op_mod = importlib.import_module(f"{pkg_name}.conv2d_3x3")
    pd_mod = importlib.import_module(f"{pkg_name}.conv2d_3x3_program_descriptor")
    # KNW_KNOBS="NAME=python-expr;...": descriptor knobs for a whole session (debug / bisect).
    for item in [k for k in os.environ.get("KNW_KNOBS", "").split(";") if k]:
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
