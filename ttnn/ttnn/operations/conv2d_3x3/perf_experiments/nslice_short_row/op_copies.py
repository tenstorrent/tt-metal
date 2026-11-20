# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load an op copy of this experiment as a package (the dirs have no __init__.py, so `import ttnn` never walks them).
Host-only: no torch, no device.

    op_base/ -> nsr_base: conv2d_3x3 at 1a04504f992 + perf_experiments/ablation_r14/ablation_hooks.patch
    op_copy/ -> nsr_copy: the same + this experiment's variants (NSR_DEFINES)
    <name>_nz -> the zone-stripped twin (mk_nz.sh)"""

import importlib
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def copy_modules(sub):
    pkg_name = "nsr_" + sub
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(HERE / sub)]
        sys.modules[pkg_name] = pkg
    return (
        importlib.import_module(f"{pkg_name}.conv2d_3x3"),
        importlib.import_module(f"{pkg_name}.conv2d_3x3_program_descriptor"),
    )
