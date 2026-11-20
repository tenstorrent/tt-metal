# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core kernel span (us) of one run of a device profile log, as a grid (rows = core_y, cols = core_x), plus
per-core zone sums. python3 -I core_map.py <profile_log_device.csv[.gz]> <run host ID> [RISC] [zone]"""
import collections
import csv
import gzip
import sys

path, run = sys.argv[1], int(sys.argv[2])
risc = sys.argv[3] if len(sys.argv) > 3 else "BRISC"
zone = sys.argv[4] if len(sys.argv) > 4 else f"{risc}-KERNEL"
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    header = f.readline()
    mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    starts = {}
    tot = collections.defaultdict(float)
    for r in rows:
        if len(r) < len(cols) or int(r[ix["run host ID"]]) != run:
            continue
        if r[ix["RISC processor type"]].strip() != risc or r[ix["zone name"]].strip() != zone:
            continue
        core = (int(r[ix["core_x"]]), int(r[ix["core_y"]]))
        t = int(r[ix["time[cycles since reset]"]])
        if r[ix["type"]].strip() == "ZONE_START":
            starts[core] = t
        elif core in starts:
            tot[core] += (t - starts.pop(core)) / mhz
xs = sorted({c[0] for c in tot})
ys = sorted({c[1] for c in tot})
print(f"{zone} on {risc}, run {run} (us)")
print("y\\x " + " ".join(f"{x:>6}" for x in xs))
for y in ys:
    print(f"{y:>3} " + " ".join(f"{tot.get((x, y), 0):6.0f}" for x in xs))
