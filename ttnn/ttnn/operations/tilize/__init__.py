# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import sys as _sys

from .tilize import (
    ARCH_HAS_FP8_TILIZE,
    EXCLUSIONS,
    INPUT_TAGGERS,
    PROPERTIES,
    SUPPORTED,
    tilize,
    validate,
)

__all__ = [
    "tilize",
    "validate",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
    "PROPERTIES",
    "ARCH_HAS_FP8_TILIZE",
]


# Bind the public top-level name. `ttnn.operations.tilize` is the load-bearing
# import path (the registry suites read INPUT_TAGGERS / SUPPORTED / EXCLUSIONS
# off it), but the externally-authored case set in
# `eval/golden_tests/tilize/test_golden_main_tests.py` calls `ttnn.tilize(...)`
# — the name this op occupies. Without the alias every one of those cases dies
# at `AttributeError: module 'ttnn' has no attribute 'tilize'` before the op is
# ever entered, which reads as 159 op failures while testing nothing.
#
# Done here rather than in `ttnn/__init__.py`: `ttnn.operations` is walked with
# `pkgutil` partway through ttnn's own import, so `sys.modules["ttnn"]` exists
# (partially initialised) by the time this module body runs, and the op package
# stays self-contained. Guarded on the module lookup so importing this package
# outside a live `ttnn` is still safe.
#
# The walk registers this package under its BARE name (`sys.modules["tilize"]`,
# see ttnn/ttnn/operations/__init__.py), so a later `from ttnn.operations.tilize
# import ...` executes the file a second time under the dotted name and that
# second instance is the one `ttnn.operations.tilize` ends up pointing at. The
# two are code-identical, but binding unconditionally means the last (dotted,
# canonical) instance wins.
_ttnn = _sys.modules.get("ttnn")
if _ttnn is not None:
    _ttnn.tilize = tilize
