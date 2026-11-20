# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tail timeline: every zone of one core (default: the core whose kernel ends last) that ends after T0 us, with no
duration filter, relative to the run's earliest kernel start.
    python3 tail_tl.py <profile_log_device.csv> <run_index (1-based among runs with >= 8 cores)> [T0] [x,y]"""
import collections
import csv
import sys

path, idx = sys.argv[1], int(sys.argv[2])
t_from = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
want = tuple(int(v) for v in sys.argv[4].split(",")) if len(sys.argv) > 4 else None
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rows:
        if len(r) < len(cols):
            continue
        runs[int(r[ix["run host ID"]])].append(
            (
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
kst, ken = {}, {}
for core, risc, t, z, kind in ev:
    if z.endswith("-KERNEL"):
        if kind == "ZONE_START":
            kst[(core, risc)] = min(kst.get((core, risc), t), t)
        else:
            ken[(core, risc)] = max(ken.get((core, risc), t), t)
t0 = min(kst.values())
us = lambda c: (c - t0) / freq
if want is None:
    want = max(ken, key=lambda k: ken[k])[0]
core_end = {}
for (c, r), v in ken.items():
    core_end[c] = max(core_end.get(c, 0), us(v))
ends = sorted(core_end.values())
print(f"run {big[idx-1]}: core end p50 {ends[len(ends)//2]:.2f} max {ends[-1]:.2f}; core {want}")
stack = collections.defaultdict(list)
out = []
for core, risc, t, z, kind in ev:
    if core != want:
        continue
    if kind == "ZONE_START":
        stack[(risc, z)].append(t)
    elif stack[(risc, z)]:
        s = stack[(risc, z)].pop()
        out.append((risc, us(s), us(t), z))
for risc in ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"):
    items = sorted([o for o in out if o[0] == risc and o[2] >= t_from], key=lambda o: o[1])
    print(f"  {risc}: " + "  ".join(f"{z}[{a:.2f}-{b:.2f}]" for _, a, b, z in items))
