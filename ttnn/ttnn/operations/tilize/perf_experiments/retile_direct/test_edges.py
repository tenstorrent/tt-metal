# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The three calls that decide the DOMAIN, re-measured deliberately:

  * the NARROWING cast (fp32 -> bf16) against a BASELINE-relative oracle. torch's
    `.to(bfloat16)` is not the packer's rounding, so the op itself is not
    bit-equal to torch here — "same bytes as the op today" is the only meaningful
    bar for the other arms.
  * `tile_h == 1`, the one geometry where the direct arms measured a REGRESSION,
    re-run so the call is not made on a single sample.
  * `tile_h == 2/4`, which brackets where that regression starts.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/retile_direct/test_edges.py
"""

import pytest
import ttnn

from ttnn.operations.tilize.perf_experiments.retile_direct import _harness as H

FOCUS = [1, 1, 1024, 1024]


def test_cast_narrowing(device):
    rows = []
    for arm in H.arms_for(32, 8):
        ns, exact = H.run(device, arm, FOCUS, 32, 8, dtype=ttnn.float32, out_dtype=ttnn.bfloat16, oracle="baseline")
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"CAST-NARROW fp32->bf16 32->8 {FOCUS} (oracle: the op's own output)")


@pytest.mark.parametrize("in_tile_h,tile_h", [(32, 1), (32, 2), (32, 4)], ids=["32to1", "32to2", "32to4"])
def test_tiny_output_tile(device, in_tile_h, tile_h):
    rows = []
    for arm in H.arms_for(in_tile_h, tile_h):
        ns, exact = H.run(device, arm, FOCUS, in_tile_h, tile_h)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"TINY-OUT bf16 {in_tile_h}->{tile_h} {FOCUS} geom={H.geometry(in_tile_h, tile_h)}")


def test_focus_repeat(device):
    """Second sample of the focus case — the arm-1 vs arm-3 ordering sat inside
    the ~2-3% noise band on the first run."""
    rows = []
    for arm in H.arms_for(32, 8):
        ns, exact = H.run(device, arm, FOCUS, 32, 8)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"FOCUS-REPEAT bf16 32->8 {FOCUS}")
