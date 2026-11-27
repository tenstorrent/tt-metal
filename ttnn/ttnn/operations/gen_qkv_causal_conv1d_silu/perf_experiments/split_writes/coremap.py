# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core end-time map (us after the program's first kernel start) from profile_log_device.csv.

    python3 coremap.py <csv> [--risc BRISC] [--zone <zone name, default <RISC>-KERNEL>] [--runs i j ...]
Prints, per run: program span, and a grid (physical NoC x across, y down) of each core's last end of the
zone; plus p50/p90/max.
"""
import csv
import sys
from collections import defaultdict

args = sys.argv[1:]
path = args[0]
risc = "BRISC"
zone = None
runs_sel = None
i = 1
while i < len(args):
    if args[i] == "--risc":
        risc = args[i + 1]
        i += 2
    elif args[i] == "--zone":
        zone = args[i + 1]
        i += 2
    elif args[i] == "--runs":
        runs_sel = [int(a) for a in args[i + 1 :]]
        break
    else:
        i += 1
zone = zone or f"{risc}-KERNEL"
f = open(path)
header = f.readline()
mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0].strip()) if "CHIP_FREQ[MHz]:" in header else 1350.0
rows = list(csv.DictReader(f, skipinitialspace=True))
by_run = defaultdict(list)
for r in rows:
    by_run[int(r["run host ID"])].append(r)
for idx, run in enumerate(sorted(by_run)):
    if runs_sel is not None and idx not in runs_sel:
        continue
    rs = by_run[run]
    starts = [
        int(r["time[cycles since reset]"])
        for r in rs
        if r["zone name"].endswith("KERNEL") and r["type"] == "ZONE_START"
    ]
    ends = [
        int(r["time[cycles since reset]"]) for r in rs if r["zone name"].endswith("KERNEL") and r["type"] == "ZONE_END"
    ]
    t0, t1 = min(starts), max(ends)
    end = {}
    for r in rs:
        if r["zone name"] == zone and r["type"] == "ZONE_END" and r["RISC processor type"] == risc:
            c = (int(r["core_x"]), int(r["core_y"]))
            end[c] = max(end.get(c, 0), (int(r["time[cycles since reset]"]) - t0) / mhz)
    xs = sorted({c[0] for c in end})
    ys = sorted({c[1] for c in end})
    v = sorted(end.values())
    print(
        f"=== run #{idx} (host id {run}) span {(t1 - t0) / mhz:.1f} us | {risc} {zone} end p50 {v[len(v)//2]:.1f} p90 {v[int(len(v)*0.9)]:.1f} max {v[-1]:.1f}"
    )
    print("y\\x " + " ".join(f"{x:5d}" for x in xs))
    for y in ys:
        print(f"{y:3d} " + " ".join(f"{end[(x, y)]:5.0f}" if (x, y) in end else "    ." for x in xs))
