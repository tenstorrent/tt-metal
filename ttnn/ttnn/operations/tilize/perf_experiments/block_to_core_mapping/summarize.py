#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Table of per-core span statistics per permutation mode (idea block_to_core_mapping).

Reads every `logs/dev_<tag>.csv` snapshot and prints, per RISC, the LAST
dispatch's per-core duration mean / max / max-over-mean, the kernel-end spread
(the wall), and the correlation of duration with grid row vs. with block id —
the one number that says whether the tail is attached to the CORE or to the
BLOCK.
"""
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json

from percore_map import collect, corr, grid_maps

D = os.path.dirname(os.path.abspath(__file__))
print(
    f"{'tag':22s} {'risc':7s} {'mean':>7s} {'max':>7s} {'max/mean':>8s} {'sum/64':>7s} {'wall':>7s} {'r(row)':>7s} {'r(blk)':>7s}"
)
for path in sorted(glob.glob(os.path.join(D, "logs", "dev_*.csv"))):
    tag = os.path.basename(path)[4:-4]
    st, _ = collect(path)
    runs = sorted({k[0] for k in st})
    if not runs:
        continue
    run = runs[-1]  # the last dispatch of the run
    mode = tag.split("_")[0]
    plan_path = os.path.join(D, "logs", f"plan_{mode}_focus.json")
    bmap = {}
    if os.path.exists(plan_path):
        recs = json.loads(open(plan_path).read())
        bmap = {(a[0], a[1]): a[2] for a in recs[0]["assignment"]}
    for risc in ("NCRISC", "BRISC"):
        got = grid_maps(st, run, risc)
        if not got:
            continue
        cells, _xs, _ys = got
        d = [v[0] for v in cells.values()]
        ends = [v[2] for v in cells.values()]
        rrow = corr([(y, cells[(x, y)][0]) for (x, y) in cells])
        rblk = corr([(bmap[c], cells[c][0]) for c in cells if c in bmap]) if bmap else float("nan")
        print(
            f"{tag:22s} {risc:7s} {statistics.mean(d):7.0f} {max(d):7.0f} {max(d)/statistics.mean(d):8.2f} "
            f"{sum(d)/len(d):7.0f} {max(ends):7.0f} {rrow:+7.2f} {rblk:+7.2f}"
        )
