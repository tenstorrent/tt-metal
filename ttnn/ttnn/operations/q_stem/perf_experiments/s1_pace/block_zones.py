# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-block stage-1 zones (QSTEM_ABLATE=BLOCK_ZONES): per chip, the last call's S1_XW (x wait) / S1_MM (matmul with
x present) instance durations on TRISC_0 (unpack) / TRISC_1 (math), median / p90 over (core, block), us. Host only.

    python3 block_zones.py [profile_log_device.csv]
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
f = open(path)
f.readline()
rows = list(csv.DictReader(f, skipinitialspace=True))
MHZ = 1350.0
last = {}
for r in rows:
    last[r["PCIe slot"]] = max(last.get(r["PCIe slot"], 0), int(r["run host ID"]))
for chip in sorted(last):
    open_ = {}
    durs = defaultdict(list)
    ends = defaultdict(list)
    t0 = None
    for r in rows:
        if r["PCIe slot"] != chip or int(r["run host ID"]) != last[chip]:
            continue
        t = int(r["time[cycles since reset]"])
        if r["zone name"].endswith("KERNEL") and r["type"] == "ZONE_START":
            t0 = t if t0 is None else min(t0, t)
        z = r["zone name"]
        if z not in ("S1_XW", "S1_MM"):
            continue
        key = (r["core_x"], r["core_y"], r["RISC processor type"], z)
        if r["type"] == "ZONE_START":
            open_[key] = t
        elif key in open_:
            durs[(r["RISC processor type"], z)].append((t - open_.pop(key)) / MHZ)
            ends[(r["RISC processor type"], z)].append(t)
    for k in sorted(durs):
        v = sorted(durs[k])
        e = sorted(ends[k])
        print(
            f"chip {chip} {k[1]:6s} {k[0]:8s} n={len(v):4d} med {v[len(v)//2]:6.2f} p90 {v[int(len(v)*.9)]:6.2f} "
            f"max {v[-1]:6.2f} sum/core {sum(v)/max(1,len(v))*20:6.1f}  last end {(e[-1]-t0)/MHZ:6.1f}"
        )
