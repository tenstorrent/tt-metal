# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core timeline for zone runs (GM_ZONES=1): per labelled op, relative to the op's earliest kernel start —
the chosen zones' (start, end) on each core (first start, last end, summed duration), each RISC's kernel end, and the
slowest cores. Use to see WHEN γ lands (wr_gamma end / cmp_wait_gamma) vs when it is needed.

    python3 <dir>/gm_zones.py [--zones wr_gamma,cmp_wait_gamma] [--top 6]
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=None)
ap.add_argument("--log", default=os.path.join(HERE, "gm_order.txt"))
ap.add_argument("--zones", default="wr_gamma,cmp_wait_gamma,cmp_wait_inv,wr_final_barrier")
ap.add_argument("--top", type=int, default=6)
a = ap.parse_args()
path = a.csv or max(glob.glob("generated/profiler/reports/*/ops_perf_results*.csv"), key=os.path.getmtime)
dev = os.path.join(os.path.dirname(path), "profile_log_device.csv")
rows = [r for r in csv.DictReader(open(path)) if "GenericOp" in r["OP CODE"]]
labels = [l.strip() for l in open(a.log) if l.strip()][-len(rows) :]
runs = {r["GLOBAL CALL COUNT"]: (lab, r) for lab, r in zip(labels, rows)}
zones = a.zones.split(",")
with open(dev) as fh:
    mhz = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    recs = [r for r in csv.DictReader(fh, skipinitialspace=True) if r["run host ID"] in runs]
by_run = defaultdict(list)
for r in recs:
    by_run[r["run host ID"]].append(r)
for run, (lab, op) in runs.items():
    rr = by_run.get(run, [])
    stack = defaultdict(list)
    kstart, kend = {}, {}
    z = defaultdict(lambda: [None, None, 0.0])  # (core, zone) -> [first start, last end, total]
    for r in rr:
        core = (int(r["core_x"]), int(r["core_y"]))
        risc, name, cyc = r["RISC processor type"], r["zone name"], int(r["time[cycles since reset]"])
        k = (core, risc, name)
        if r["type"] == "ZONE_START":
            stack[k].append(cyc)
            continue
        if r["type"] != "ZONE_END" or not stack[k]:
            continue
        st = stack[k].pop()
        if name.endswith("-KERNEL"):
            kstart[(core, risc)] = st
            kend[(core, risc)] = cyc
        elif name in zones:
            e = z[(core, name)]
            e[0] = st if e[0] is None else min(e[0], st)
            e[1] = cyc if e[1] is None else max(e[1], cyc)
            e[2] += cyc - st
    if not kstart:
        continue
    t0 = min(kstart.values())
    us = lambda c: (c - t0) / mhz  # cycles -> us since op start
    print(f"\n## {lab}  ({float(op['DEVICE KERNEL DURATION [ns]']):.0f} ns)")
    ends = defaultdict(dict)
    for (core, risc), e in kend.items():
        ends[core][risc] = us(e)
    order = sorted(ends, key=lambda c: -max(ends[c].values()))
    for name in zones:
        vals = [(c, v) for (c, n), v in z.items() if n == name]
        if not vals:
            continue
        tot = [v[2] / mhz for _, v in vals]
        last = max(us(v[1]) for _, v in vals)
        print(
            f"  {name:<16} cores {len(vals):>3}  total/core mean {sum(tot) / len(tot):7.2f} max {max(tot):7.2f} us"
            f"  last end {last:7.2f} us"
        )
    print(f"  slowest cores (end us per RISC; zone [start,end] us):")
    for c in order[: a.top] + order[-2:]:
        zs = "  ".join(
            f"{n}[{us(z[(c, n)][0]):.1f},{us(z[(c, n)][1]):.1f}]" for n in zones if (c, n) in z and z[(c, n)][0]
        )
        e = "  ".join(f"{r[:2]}{v:7.2f}" for r, v in sorted(ends[c].items()))
        print(f"    core{c}: {e}  | {zs}")
