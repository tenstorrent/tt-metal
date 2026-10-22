# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared import path. The body stays in tt_lib._internal.comparison_funcs so the wheel stays self-contained.

The file is loaded by path. Importing ``tt_lib`` runs ``tt_lib/__init__.py``, which imports ``ttnn``, and with
``PYTHONPATH`` set to the repo root that import resolves to the installed copy rather than this checkout. The
historical sweep path was a symlink to this file and did not import ``tt_lib``.
"""

import importlib.util
import sys
from pathlib import Path

_impl_path = Path(__file__).resolve().parents[1] / "ttnn" / "tt_lib" / "_internal" / "comparison_funcs.py"
_spec = importlib.util.spec_from_file_location(__name__, _impl_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load comparison helpers from {_impl_path}")
_impl = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _impl
_spec.loader.exec_module(_impl)
