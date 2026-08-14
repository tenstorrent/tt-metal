# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage zone breakdown of the baseline vs the recommended arm.

Answers "what is left?" after the win: if the whole reader collapses into a NoC
wait, the retile has arrived at the same DRAM bound the row-major path runs at
and there is no second round to play here.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/retile_permute/test_breakdown.py
"""

import pytest
from loguru import logger

from ttnn.operations.tilize.perf_experiments import _zones
from ttnn.operations.tilize.perf_experiments.retile_permute import _harness as H

FOCUS = [1, 1, 1024, 1024]


@pytest.mark.parametrize(
    "in_tile_h,tile_h,variant",
    [(32, 8, 0), (32, 8, 4), (1, 32, 0), (1, 32, 2)],
    ids=["32to8_baseline", "32to8_direct_dram", "1to32_baseline", "1to32_rm_stage_direct"],
)
def test_breakdown(device, in_tile_h, tile_h, variant):
    _zones.clear()
    ns, exact = H.run(device, variant, FOCUS, in_tile_h, tile_h)
    assert exact
    stages, diag = _zones.breakdown()
    freq = device.get_clock_rate_mhz() if hasattr(device, "get_clock_rate_mhz") else 1000.0
    tensor_bytes = 2
    for d in FOCUS:
        tensor_bytes *= d
    logger.info(
        f"\n=== {in_tile_h}->{tile_h} arm {variant}:{H.VARIANTS[variant][0]}  whole-op {ns:.0f} ns "
        f"({2 * tensor_bytes / ns:.1f} GB/s) ===\n" + _zones.report(stages, diag, freq)
    )
