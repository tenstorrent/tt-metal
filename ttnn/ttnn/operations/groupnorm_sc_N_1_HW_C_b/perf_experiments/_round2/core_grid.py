#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core grid map of one zone's per-core duration (us) for one run: rows = physical core_y, cols = core_x.
usage: core_grid.py <profile_log_device.csv> <run host ID> <RISC> <zone> [<zone2> ...]   (zones summed)
"""
import csv, sys
from collections import defaultdict

path, run, risc, zones = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
with open(path) as f:
    head = f.readline()
    mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = [
        r
        for r in csv.DictReader(f, skipinitialspace=True)
        if r["run host ID"] == run and r["RISC processor type"] == risc
    ]
k = 1.0 / mhz
st = defaultdict(list)
dur = defaultdict(float)
for r in rows:
    if r["zone name"] not in zones or r["type"] not in ("ZONE_START", "ZONE_END"):
        continue
    c = (int(r["core_x"]), int(r["core_y"]))
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        st[(c, r["zone name"])].append(t)
    elif st[(c, r["zone name"])]:
        dur[c] += (t - st[(c, r["zone name"])].pop()) * k
xs = sorted({c[0] for c in dur})
ys = sorted({c[1] for c in dur})
print("y\\x " + " ".join(f"{x:6d}" for x in xs))
for y in ys:
    print(f"{y:3d} " + " ".join(f"{dur[(x, y)]:6.0f}" if (x, y) in dur else "     -" for x in xs))
