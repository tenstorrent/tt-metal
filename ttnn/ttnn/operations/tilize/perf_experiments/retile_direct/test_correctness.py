# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness gate for every arm — run this FIRST. A retile is a pure
permutation (plus at most a value-preserving cast), so the bar is BIT-EXACT
against the torch input, not a PCC.

Nothing here measures. An arm that is fast and wrong is `incorrect`, not a win.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/retile_direct/test_correctness.py
"""

import pytest
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.retile_direct import _harness as H

SMALL = [1, 1, 256, 256]

GEOMS = [(32, 8), (32, 16), (32, 1), (1, 32), (8, 32), (16, 32)]


@pytest.mark.parametrize("arm", sorted(H.VARIANTS), ids=[H.VARIANTS[a][0] for a in sorted(H.VARIANTS)])
@pytest.mark.parametrize("in_tile_h,tile_h", GEOMS, ids=[f"{a}to{b}" for a, b in GEOMS])
def test_geometry(device, in_tile_h, tile_h, arm):
    if arm not in H.arms_for(in_tile_h, tile_h):
        pytest.skip("arm compiles to another arm on this geometry")
    _ns, exact = H.run(device, arm, SMALL, in_tile_h, tile_h, measure=False)
    g = H.geometry(in_tile_h, tile_h)
    logger.info(f"CORRECTNESS {in_tile_h}->{tile_h} arm={H.VARIANTS[arm][0]} exact={exact} geom={g}")
    assert exact, f"arm {H.VARIANTS[arm][0]} not bit-exact on {in_tile_h}->{tile_h} (geom {g})"


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.uint8], ids=["fp32", "uint8"])
@pytest.mark.parametrize("in_tile_h,tile_h", [(32, 8), (8, 32)], ids=["32to8", "8to32"])
def test_dtype(device, dtype, in_tile_h, tile_h):
    rows = []
    for arm in H.arms_for(in_tile_h, tile_h, dtype):
        _ns, exact = H.run(device, arm, SMALL, in_tile_h, tile_h, dtype=dtype, measure=False)
        rows.append((arm, H.VARIANTS[arm][0], exact))
    g = H.geometry(in_tile_h, tile_h, dtype)
    logger.info(f"CORRECTNESS {dtype} {in_tile_h}->{tile_h} geom={g} " + str(rows))
    # NOT asserted: the point of this cell is to FIND where an arm is incorrect
    # (uint8 at rows_per_run==1 makes the direct-DRAM run 16 B on a 32 B-aligned
    # DRAM). The table is the deliverable.


def test_cast_retile(device):
    """The case the plain direct arms cannot express: a retile that also CASTS.
    Arms 1/2/5/6 hand raw INPUT bytes to the writer, so nobody owns the packer's
    conversion; arms 3/4 route the permuted tile through a compute datacopy."""
    rows = []
    for arm in H.arms_for(32, 8):
        _ns, exact = H.run(device, arm, SMALL, 32, 8, dtype=ttnn.bfloat16, out_dtype=ttnn.float32, measure=False)
        rows.append((arm, H.VARIANTS[arm][0], exact))
    logger.info("CAST-RETILE bf16->fp32 32->8: " + str(rows))
    for arm, slug, exact in rows:
        if arm in (0, 3, 4):
            assert exact, f"the cast-capable arm {slug} must be bit-exact on a casting retile"
