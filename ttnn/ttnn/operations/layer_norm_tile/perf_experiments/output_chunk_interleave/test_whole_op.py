# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole-op A/B: LN_OCI=<real|base|raw_g12|rawskew_g8|...> scripts/run_safe_pytest.sh [--profile] <this file>.

real = the unmodified op; anything else runs pd_oci + op_kernels with that output-stage variant.
"""

import pytest

import install_oci  # noqa: F401  (patches the op before use)

from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_perf_bench import GUARD_CASES, IDS, PERF_CASES

ROWS = dict(zip(IDS, PERF_CASES))
ROWS.update(GUARD_CASES)
# Peel rows first (focus1..3, 8x384, 32x32), then the remaining perf rows and the guard rows.
PEEL = [n for n in ROWS if n.startswith(("focus", "8x384", "32x32"))]
ORDER = PEEL + [n for n in ROWS if n not in PEEL]


@pytest.mark.parametrize("name", ORDER)
def test_row(device, name):
    case = dict(ROWS[name])
    inputs = case.pop("inputs")
    run_layer_norm(inputs, device=device, **case)
