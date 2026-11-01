# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""work_split_pipeline bench: peel rows + guard rows through the descriptor copy (load with -p ws_swap)."""

import os

import pytest

import ws_descriptor
from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_perf_bench import GUARD_CASES, IDS, PERF_CASES

ROWS = {name: case for name, case in zip(IDS, PERF_CASES)}
ROWS.update(GUARD_CASES)
SEL = os.environ.get("WS_ROWS")
if SEL:
    ROWS = {name: ROWS[name] for name in SEL.split(",")}


@pytest.mark.parametrize("case", list(ROWS.values()), ids=list(ROWS))
def test_ws(device, case):
    assert ws_descriptor.SPLIT == os.environ.get("WS_SPLIT", "base")
    case = dict(case)
    inputs = case.pop("inputs")
    run_layer_norm(inputs, device=device, **case)
