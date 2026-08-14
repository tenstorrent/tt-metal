# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The measured bake-off on the FOCUS case. Run test_correctness.py FIRST —
nothing here asserts a perf direction, and a wrong arm is disqualified there.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/retile_direct/test_focus.py

ONE fresh-cache measured launch per arm per case (device kernel duration has no
warm-up transient, so a trial loop would re-measure the same number N times).
Box/arch: bgd-lab-16, wormhole_b0.
"""

import pytest
import ttnn

from ttnn.operations.tilize.perf_experiments.retile_direct import _harness as H

FOCUS = [1, 1, 1024, 1024]


def test_focus_32to8(device):
    """The coordinator's focus shape: [1,1,1024,1024] bf16, 32 -> 8, DRAM->DRAM."""
    rows = []
    for arm in H.arms_for(32, 8):
        ns, exact = H.run(device, arm, FOCUS, 32, 8)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"FOCUS bf16 32->8 {FOCUS}")


def test_focus_cast(device):
    """The SAME focus retile, but bf16 -> fp32. This is the case the pure-direct
    arms cannot express (test_correctness proves them wrong here); what it costs
    to keep the direct byte move and let COMPUTE own the cast is the point."""
    rows = []
    for arm in H.arms_for(32, 8):
        ns, exact = H.run(device, arm, FOCUS, 32, 8, dtype=ttnn.bfloat16, out_dtype=ttnn.float32)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"FOCUS-CAST bf16->fp32 32->8 {FOCUS}")


@pytest.mark.parametrize("shape", [[1, 1, 256, 256], [1, 1, 2048, 2048]], ids=["small", "4x"])
def test_tile_count(device, shape):
    """Does the pattern hold when there is barely any work, and at 4x the work?"""
    rows = []
    for arm in H.arms_for(32, 8):
        ns, exact = H.run(device, arm, shape, 32, 8)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"TILE-COUNT bf16 32->8 {shape}")
