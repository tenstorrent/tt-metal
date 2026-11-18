# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""test_kwfs_unit.py (conv2d_3x3's unit tests on this experiment's copy), recording which tests take kw-fold
(kw_fold_selected.txt next to this file).

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_seedless/test_kwfs_unit_sel.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kwfs_copy import copy_modules  # noqa: E402
from test_kwfs_unit import *  # noqa: E402,F401,F403

_LOG = Path(__file__).resolve().parent / "kw_fold_selected.txt"


@pytest.fixture(autouse=True)
def _record_kw_fold(request):
    _, pd = copy_modules()
    orig = pd.kw_fold_mode

    def rec(*a):
        v = orig(*a)
        if v:
            with open(_LOG, "a") as f:
                f.write(f"{request.node.nodeid} kblocked={pd.kw_fold_kblocked(a[0])}\n")
        return v

    pd.kw_fold_mode = rec
    yield
    pd.kw_fold_mode = orig
