# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""DOMAIN sweep — the same core-count arms on the regimes that already have
several blocks per core, plus the smallest regime.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/oneblock_overlap/test_sweep.py

These are the shapes a core-count rule must NOT regress: (a) and (c) already sit
at ~4 blocks/core, so halving the grid there doubles blocks/core while HALVING
the aggregate DRAM concurrency — that is the trade the sweep prices. (d) is the
smallest regime, where the split already lights only 2 cores and the cap is
structurally inert.
"""

import pytest
import ttnn

from ._harness import bake_off

_SWEEP_ARMS = ["baseline", "cores32", "cores32_d2_notrid", "cores16"]


@pytest.mark.parametrize("shape_key", ["a_square", "c_multiblock", "d_smallest"])
def test_sweep(device, shape_key):
    bake_off(device, shape_key, _SWEEP_ARMS, rounds=5, dtype=ttnn.bfloat16)
