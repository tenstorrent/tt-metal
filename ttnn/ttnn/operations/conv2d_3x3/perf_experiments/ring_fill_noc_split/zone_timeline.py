# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Zone executions on a few cores, as <start us from kernel start>+<duration us>:
zone_timeline.py <profile_log_device.csv[.gz]> <run ids, comma> <x,y;x,y;...> <zone,zone,...> [max per row]"""
import collections
import csv
import gzip
import sys

path, runs, cores, zones = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else 40
runs = {int(r) for r in runs.split(",")}
want = {tuple(int(v) for v in c.split(",")) for c in cores.split(";")}
zones = set(zones.split(","))
kern = {"NCRISC-KERNEL", "BRISC-KERNEL"}
ev = collections.defaultdict(list)
with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
    f.readline()
    r = csv.reader(f)
    cols = [c.strip() for c in next(r)]
    ix = {c: i for i, c in enumerate(cols)}
    for row in r:
        if len(row) < len(cols):
            continue
        run = int(row[ix["run host ID"]])
        if run not in runs:
            continue
        k = (int(row[ix["core_x"]]), int(row[ix["core_y"]]))
        if k not in want:
            continue
        z = row[ix["zone name"]].strip()
        if z not in zones and z not in kern:
            continue
        ev[(run, k, z)].append((int(row[ix["time[cycles since reset]"]]), row[ix["type"]].strip()))
start = {}
for (run, k, z), lst in ev.items():
    if z == "NCRISC-KERNEL":
        start[(run, k)] = min(t for t, _ in lst)
for (run, k, z), lst in sorted(ev.items()):
    if z in kern and z not in zones:
        continue
    lst.sort()
    out, st, tot = [], None, 0.0
    for t, ty in lst:
        if ty == "ZONE_START":
            st = t
        elif st is not None:
            d = (t - st) / 1350
            tot += d
            out.append(f"{(st - start[(run, k)]) / 1350:.0f}+{d:.1f}")
            st = None
    print(run, k, z, f"n={len(out)} sum={tot:.0f}", " ".join(out[:limit]))
