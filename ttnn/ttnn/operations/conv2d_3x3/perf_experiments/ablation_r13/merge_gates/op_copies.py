# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load an op copy as a package (no __init__.py: `import ttnn` must not walk them). Host-only, no torch, no device.

    op_base/ -> package r13_base: conv2d_3x3 at 4f2f03f6d20 (end of Perf 12)
    op_copy/ -> package r13_copy: the tree under test (Perf 13 final)"""

import importlib
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGS = {"copy": ("r13_copy", "op_copy"), "base": ("r13_base", "op_base")}


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
