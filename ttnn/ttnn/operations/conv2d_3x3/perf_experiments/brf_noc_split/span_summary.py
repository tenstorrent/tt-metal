# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-grid-row / per-grid-column NCRISC kernel span (us) of one program run in a device profile:
span_summary.py <profile_log_device.csv> <run host ID> [zone]   (zone default NCRISC-KERNEL)."""
import collections
import csv
import sys

path, run = sys.argv[1], int(sys.argv[2])
zone = sys.argv[3] if len(sys.argv) > 3 else "NCRISC-KERNEL"
st, dur = {}, collections.defaultdict(float)
with open(path) as f:
    f.readline()
    r = csv.reader(f)
    cols = [c.strip() for c in next(r)]
    ix = {c: i for i, c in enumerate(cols)}
    for row in r:
        if len(row) < len(cols) or int(row[ix["run host ID"]]) != run or row[ix["zone name"]].strip() != zone:
            continue
        k = (int(row[ix["core_x"]]), int(row[ix["core_y"]]))
        t = int(row[ix["time[cycles since reset]"]])
        if row[ix["type"]].strip() == "ZONE_START":
            st[k] = t
        else:
            dur[k] += (t - st.pop(k)) / 1350.0
rows = collections.defaultdict(list)
colv = collections.defaultdict(list)
for (x, y), d in dur.items():
    rows[y].append(d)
    colv[x].append(d)
print("row max: " + " ".join(f"y{y}={max(v):.0f}" for y, v in sorted(rows.items())))
print("col max: " + " ".join(f"x{x}={max(v):.0f}" for x, v in sorted(colv.items())))
allv = list(dur.values())
print(f"cores {len(allv)}  max {max(allv):.0f}  mean {sum(allv) / len(allv):.0f}  min {min(allv):.0f}")
