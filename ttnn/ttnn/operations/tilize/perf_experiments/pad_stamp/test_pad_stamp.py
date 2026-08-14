# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf bake-off: the writer-side OUTPUT-format pad stamp.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/pad_stamp/test_pad_stamp.py

ARMS (writer kernel only; identical program otherwise)
    real            the op as it ships (harness validation for v0)
    v0_baseline     the op's current stamp, reconstructed verbatim in this dir
    v1_unroll       option 2 — widen/unroll the fill's store loop
    v2_replicate    option 1 — stamp one tile, replicate it into the CB (L1->L1)
    v3_srcsub       whole-pad pages written straight from the pre-stamped tile
    v4_fused        v1 + v3
    v5_slotcache    option 3 — "the slot already holds the pattern". MEASURED
                    INCORRECT (1,048,576 / 4,194,304 padded positions wrong on
                    the focus case, first at row 1088 — the first whole-pad row
                    after the two CB slots start recycling). Compute repacks
                    every byte of the slot each block, so nothing about a slot's
                    previous contents survives. Kept as the disqualifying
                    evidence, not asserted as a pass.
    ceiling         the op's own out_fill=0 arm: NOT a candidate (numerically
                    wrong here — the fill is inexact in the input dtype), only an
                    upper bound on what deleting the stamp could ever buy.

GEOMETRIES — chosen so the domain claim is not read off one shape:
    focus        [1,1,1024,2048] -> [1,1,2048,2048]  half the tile-rows are WHOLE
                 pad tiles, no W tail at all (the assigned focus case)
    mixed        [1,1,1000,1000] -> [1,1,2048,2048]  whole-pad tiles AND both
                 ragged tails in the same program
    ragged       [1,1,50,50]     -> [1,1,64,64]      NO whole-pad tile exists —
                 option 1/3's "replicate an identical tile" premise is absent
    shard_pad    [1,1,1020,256]  -> [1,1,2048,256] HEIGHT x8 — whole-pad tiles on
                 the zero-copy shard placement (no outgoing write to re-source)
    shard_tail   [1,1,2040,256]  -> [1,1,2048,256] HEIGHT x8 — H tail only
    tiny_tile    16-row output tiles (Refinement 5 geometry)
    multi_img    rank-4, per-IMAGE H pad (the `img` branch of the geometry)
    dbuf_off     double buffering off => the writer's non-B8 issue loop

MEASURED, Wormhole B0, one fresh-cache run per arm (whole-op ns, and the
writer_stamp zone in ns/core):

    geometry     v0_baseline        v3_srcsub          speedup
    focus        386,749 (172,896)  142,761 ( 3,696)   2.71x
    mixed        394,664 (263,452)  139,854 ( 9,836)   2.82x
    tiny_tile    395,506 (263,067)  131,761 ( 6,221)   3.00x
    dbuf_off     394,194 (258,538)  136,776 ( 8,766)   2.88x
    shard_pad    366,475 (182,542)   22,487 ( 9,925)  16.3x
    multi_img     16,105 (  6,932)   11,176 ( 4,123)   1.44x
    ragged         6,800 (  2,324)    6,688 ( 2,322)   flat (no whole-pad tile)
    shard_tail    28,251 (  2,139)   28,121 ( 2,331)   flat (no whole-pad tile)

    focus/v1_unroll    385,495 — NULL (the fill is store-bound, not loop-bound)
    focus/v2_replicate 142,520 — tied with v3 at the whole-op level
    focus/ceiling      188,301 — out_fill=0 is SLOWER than the candidates: it
                       also switches the READER's input-format fill back on.
"""

from __future__ import annotations

import pytest
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.pad_stamp import _harness as H

CASES = {
    "focus": H.Case("focus", [1, 1, 1024, 2048], [1, 1, 2048, 2048]),
    "mixed": H.Case("mixed", [1, 1, 1000, 1000], [1, 1, 2048, 2048]),
    "ragged": H.Case("ragged", [1, 1, 50, 50], [1, 1, 64, 64]),
    "shard_pad": H.Case("shard_pad", [1, 1, 1020, 256], [1, 1, 2048, 256], shard=8),
    "shard_tail": H.Case("shard_tail", [1, 1, 2040, 256], [1, 1, 2048, 256], shard=8),
    # tile geometry: a TINY (16-row) output tile — the scratch page and the
    # whole-pad test are tile-height driven, so this is the geometry that would
    # break a 32-row assumption.
    "tiny_tile": H.Case("tiny_tile", [1, 1, 1000, 1000], [1, 1, 2048, 2048], tile_h=16),
    # rank-4 with several images: whole-pad tile-rows arise from the PER-IMAGE
    # H pad (the `img` / nth_per_img branch), not from a trailing pad region.
    "multi_img": H.Case("multi_img", [1, 4, 50, 256], [1, 4, 128, 256]),
    # double buffering off => the writer takes its NON-B8 loop (write_trid=0 on
    # an accessor output), the other of the two issue paths the substitution
    # touches.
    "dbuf_off": H.Case("dbuf_off", [1, 1, 1000, 1000], [1, 1, 2048, 2048], double_buffer=False),
}

RESULTS = []


def _record(name, arm, ns, stages, ok):
    RESULTS.append((name, arm, ns, stages.get("writer_stamp", float("nan")), ok))
    logger.info(
        "PAD_STAMP TABLE " + " | ".join(f"{r[0]}/{r[1]}: {r[2]:.0f} ns (stamp {r[3]:.0f})" for r in RESULTS[-1:])
    )


# --- focus geometry: every arm ---------------------------------------------
@pytest.mark.parametrize("arm", ["real", "v0_baseline", "v1_unroll", "v2_replicate", "v3_srcsub", "v4_fused"])
def test_focus(device, arm):
    case = CASES["focus"]
    ns, stages, ok, report = H.run(device, case, None if arm == "real" else arm)
    _record("focus", arm, ns, stages, ok)
    assert ok, report


def test_focus_ceiling_out_fill_off(device):
    """NOT a candidate: `out_fill=0` leaves the pad input-rounded (10.1875 in an
    fp32 tensor). Measured only as the upper bound on deleting the stamp."""
    case = CASES["focus"]
    ns, stages, ok, _ = H.run(device, case, None, check=False, levers=dict(out_fill=0), label="focus/ceiling")
    _record("focus", "ceiling(out_fill=0, WRONG)", ns, stages, False)


def test_focus_slotcache_is_incorrect(device):
    """Option 3 as assigned. Compute repacks every byte of the CB slot each
    block, so a 'this slot already holds the pattern' cache reads stale tiles.
    Recorded as EVIDENCE, not asserted as a pass."""
    case = CASES["focus"]
    ns, stages, ok, report = H.run(device, case, "v5_slotcache", label="focus/v5_slotcache")
    _record("focus", "v5_slotcache", ns, stages, ok)
    logger.info(f"PAD_STAMP slotcache correctness={ok} report={report}")


# --- domain sweep -----------------------------------------------------------
@pytest.mark.parametrize(
    "geometry", ["mixed", "ragged", "shard_pad", "shard_tail", "tiny_tile", "multi_img", "dbuf_off"]
)
@pytest.mark.parametrize("arm", ["real", "v0_baseline", "v1_unroll", "v2_replicate", "v3_srcsub", "v4_fused"])
def test_domain(device, geometry, arm):
    case = CASES[geometry]
    ns, stages, ok, report = H.run(device, case, None if arm == "real" else arm)
    _record(geometry, arm, ns, stages, ok)
    assert ok, report


def test_zz_summary(device):
    lines = [f"{n:12s} {a:26s} {ns:10.0f} ns   stamp {st:9.0f} ns/core   correct={ok}" for n, a, ns, st, ok in RESULTS]
    logger.info("\n=== PAD_STAMP SUMMARY ===\n" + "\n".join(lines))
