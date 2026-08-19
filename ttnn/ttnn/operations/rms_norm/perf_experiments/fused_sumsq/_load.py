# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Path-based module loader for this experiment dir.

`perf_experiments/` is a package whose every module gets exec'd by
`ttnn/ttnn/operations/__init__.py` (it walks the whole tree at `import ttnn`).
This dir deliberately has NO `__init__.py`, so nothing here is auto-imported and a
half-written file cannot break `import ttnn` for the sibling experiments sharing
this checkout.  The price is that intra-dir imports go through here instead of
relative import syntax.
"""

import importlib.util
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def load(name):
    key = f"_fused_sumsq_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod
