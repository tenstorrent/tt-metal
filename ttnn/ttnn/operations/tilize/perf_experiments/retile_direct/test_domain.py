# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Domain sweep: the axes the focus case does not move, so the carve-outs are
MEASURED rather than assumed.

  * GEOMETRY — the run length of the landed permutation is
    min(out_face_h, src_face_h) * 16 * elem_bytes, so both tile heights move the
    transfer size directly. 32->1 and 1->32 are the 32 B floor; 32->16 / 16->32
    are the two geometries where the FULL-WIDTH merge arms exist at all.
  * DTYPE — fp32 doubles the run, uint8 halves it. uint8 at a 1-row face is the
    ONLY cell on wormhole_b0 where the DRAM-direct run (16 B) falls below the
    32 B DRAM NoC alignment — the alignment carve-out, measured not assumed.
  * SHARDED OUTPUT — a resident L1 output shard takes W_REGION work assignment
    and, for the direct arms, ALIASES the CB the reader writes onto the output
    tensor itself (the writer then has nothing to move).

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/retile_direct/test_domain.py
"""

import pytest
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.retile_direct import _harness as H

FOCUS = [1, 1, 1024, 1024]

GEOMS = [(32, 16), (32, 4), (32, 1), (1, 32), (8, 32), (16, 32)]


@pytest.mark.parametrize("in_tile_h,tile_h", GEOMS, ids=[f"{a}to{b}" for a, b in GEOMS])
def test_geometry(device, in_tile_h, tile_h):
    rows = []
    for arm in H.arms_for(in_tile_h, tile_h):
        ns, exact = H.run(device, arm, FOCUS, in_tile_h, tile_h)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"GEOM bf16 {in_tile_h}->{tile_h} {FOCUS} geom={H.geometry(in_tile_h, tile_h)}")


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.uint8], ids=["fp32", "uint8"])
@pytest.mark.parametrize("in_tile_h,tile_h", [(32, 8), (8, 32)], ids=["32to8", "8to32"])
def test_dtype(device, dtype, in_tile_h, tile_h):
    rows = []
    for arm in H.arms_for(in_tile_h, tile_h, dtype):
        ns, exact = H.run(device, arm, FOCUS, in_tile_h, tile_h, dtype=dtype)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"DTYPE {dtype} {in_tile_h}->{tile_h} geom={H.geometry(in_tile_h, tile_h, dtype)}")


@pytest.mark.parametrize("in_tile_h,tile_h", [(32, 1), (1, 32)], ids=["32to1", "1to32"])
def test_uint8_align(device, in_tile_h, tile_h):
    """THE alignment carve-out. uint8 with a one-row face makes the DRAM-direct
    run 16 B, below wormhole_b0's 32 B DRAM NoC alignment. Correctness is the
    question here, not speed."""
    rows = []
    for arm in H.arms_for(in_tile_h, tile_h, ttnn.uint8):
        ns, exact = H.run(device, arm, [1, 1, 512, 512], in_tile_h, tile_h, dtype=ttnn.uint8)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    g = H.geometry(in_tile_h, tile_h, ttnn.uint8)
    H.table(rows, f"UINT8-ALIGN {in_tile_h}->{tile_h} geom={g}")
    logger.info(f"UINT8-ALIGN predicate says dram_aligned={g['dram_aligned']}; measured " + str(rows))


@pytest.mark.parametrize("in_tile_h,tile_h", [(32, 8), (1, 32)], ids=["32to8", "1to32"])
def test_sharded_output(device, in_tile_h, tile_h):
    """A resident L1 output shard: W_REGION work assignment, and for the direct
    arms the CB the reader writes IS the output tensor."""
    shape = [1, 1, 1024, 256]
    cfg = H.height_shard(shape, 8)
    rows = []
    for arm in H.arms_for(in_tile_h, tile_h):
        ns, exact = H.run(device, arm, shape, in_tile_h, tile_h, out_mem_config=cfg)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"SHARDED-OUT bf16 {in_tile_h}->{tile_h} {shape}")


def test_cast_narrowing(device):
    """The OTHER cast direction: fp32 -> bf16 (narrowing). The direct arms move
    INPUT-format bytes, so the intermediate CB page is the WIDER one here."""
    rows = []
    for arm in H.arms_for(32, 8):
        ns, exact = H.run(device, arm, FOCUS, 32, 8, dtype=ttnn.float32, out_dtype=ttnn.bfloat16)
        rows.append((arm, H.VARIANTS[arm][0], ns, exact))
    H.table(rows, f"CAST-NARROW fp32->bf16 32->8 {FOCUS}")
