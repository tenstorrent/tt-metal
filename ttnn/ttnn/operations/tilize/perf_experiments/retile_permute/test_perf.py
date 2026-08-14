# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The measured bake-off. Run test_correctness.py FIRST — nothing here asserts a
perf direction, and a wrong arm is disqualified by that file, not by this one.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/retile_permute/test_perf.py

ONE fresh-cache measured launch per arm per case (device kernel duration has no
warm-up transient, so a trial loop would re-measure the same number N times).
"""

import pytest
from loguru import logger

from ttnn.operations.tilize.perf_experiments.retile_permute import _harness as H

FOCUS = [1, 1, 1024, 1024]

# The three headline retiles (the coordinator's Step-1 cases) plus the domain
# sweep: both directions, several heights, and a second tile count.
CASES = [
    (FOCUS, 32, 8),
    (FOCUS, 32, 16),
    (FOCUS, 1, 32),
    (FOCUS, 32, 4),
    (FOCUS, 8, 32),
    (FOCUS, 16, 32),
    ([1, 1, 256, 256], 32, 8),  # small: is the pattern flat when there is no work?
    ([1, 1, 2048, 2048], 32, 8),  # 4x the tile count
]
IDS = [f"{s[-2]}x{s[-1]}_{a}to{b}" for s, a, b in CASES]


@pytest.mark.parametrize("shape,in_tile_h,tile_h", CASES, ids=IDS)
def test_bakeoff(device, shape, in_tile_h, tile_h):
    rows = []
    for variant in H.arms_for(in_tile_h):
        ns, exact = H.run(device, variant, shape, in_tile_h, tile_h)
        rows.append((variant, H.VARIANTS[variant][0], ns, exact))
    base = next(ns for v, _s, ns, _e in rows if v == 0)
    table = [f"=== BAKEOFF {shape} {in_tile_h}->{tile_h}  (baseline {base:.0f} ns) ==="]
    for variant, slug, ns, exact in sorted(rows, key=lambda r: r[2]):
        table.append(f"  {variant} {slug:16s} {ns:10.0f} ns   x{base / ns:5.2f}   bit_exact={exact}")
    logger.info("\n".join(table))
