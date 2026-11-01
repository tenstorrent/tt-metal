# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""affine_receive_overlap whole-op A/B (use with -p swap; LN_VARIANT selects kernels/<variant>)."""

import pytest

from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_perf_bench import GUARD_CASES, IDS, PERF_CASES

ALL = dict(zip(IDS, PERF_CASES))
PEEL = {name: case for name, case in ALL.items() if name.startswith(("focus", "8x384", "32x32"))}
REST = {name: case for name, case in ALL.items() if name not in PEEL}


def _run(device, case):
    case = dict(case)
    inputs = case.pop("inputs")
    run_layer_norm(inputs, device=device, **case)


@pytest.mark.parametrize("name", list(PEEL))
def test_peel(device, name):
    _run(device, PEEL[name])


@pytest.mark.parametrize("name", list(REST) + list(GUARD_CASES))
def test_guard(device, name):
    _run(device, {**REST, **GUARD_CASES}[name])
