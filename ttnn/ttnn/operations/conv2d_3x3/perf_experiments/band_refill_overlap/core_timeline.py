# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Zone timeline of one core in one run: start (us from the core's BRISC kernel start), duration (us), per zone.
python3 -I core_timeline.py <profile_log_device.csv[.gz]> <run> <x> <y> [zone,zone...]"""
import csv
import gzip
import sys

path, run, cx, cy = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
want = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else None
op = gzip.open if path.endswith(".gz") else open
ev = []
with op(path, "rt") as f:
    header = f.readline()
    mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    open_ = {}
    for r in rows:
        if len(r) < len(cols) or int(r[ix["run host ID"]]) != run:
            continue
        if (int(r[ix["core_x"]]), int(r[ix["core_y"]])) != (cx, cy):
            continue
        risc, z, ty = r[ix["RISC processor type"]].strip(), r[ix["zone name"]].strip(), r[ix["type"]].strip()
        t = int(r[ix["time[cycles since reset]"]])
        k = (risc, z)
        if ty == "ZONE_START":
            open_[k] = t
        elif k in open_:
            ev.append((open_.pop(k), t, risc, z))
t0 = min(s for s, e, r, z in ev if z == "BRISC-KERNEL")
for s, e, r, z in sorted(ev):
    if want and z not in want:
        continue
    print(f"{(s - t0) / mhz:9.1f} {(e - s) / mhz:8.1f}  {r:8s} {z}")
