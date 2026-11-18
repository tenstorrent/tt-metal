# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core phase ends for one run (no device): python3 cores.py <device.csv> <run index> [n]
Times in us from the run's earliest marker: untilize end, tilize start/end, walk start/end, kernel end."""
import collections
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
import zone_report as zr

path, idx = sys.argv[1], int(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
freq, runs = zr.load(path)
run = sorted(runs)[idx]
ev = runs[run]
t0 = min(e[2] for e in ev)
per = collections.defaultdict(dict)
stack = collections.defaultdict(list)
for c, risc, t, zone, kind in sorted(ev, key=lambda e: e[2]):
    if kind == "ZONE_START":
        stack[(c, risc, zone)].append(t)
    elif kind == "ZONE_END" and stack[(c, risc, zone)]:
        s = stack[(c, risc, zone)].pop()
        d = per[c]
        key = zone
        if key not in d:
            d[key] = [(s - t0) / freq, (t - t0) / freq]
        else:
            d[key][1] = (t - t0) / freq
rows = []
for c, d in per.items():
    kend = max(v[1] for k, v in d.items() if k.endswith("-KERNEL"))
    g = lambda z, i: d.get(z, [float("nan")] * 2)[i]
    rows.append(
        (
            kend,
            c,
            g("compute_untilize", 1),
            g("compute_tilize", 0),
            g("compute_tilize", 1),
            g("compute_matmul_walk", 0),
            g("compute_matmul_walk", 1),
            g("reader_fill", 1),
            g("reader_center", 1),
        )
    )
rows.sort(reverse=True)
print("core      kend  untl_e  tlz_s  tlz_e  walk_s walk_e fill_e ctr_e")
for r in rows[:n]:
    print(f"{str(r[1]):9s} " + " ".join(f"{x:6.2f}" for x in (r[0],) + r[2:]))
import statistics

print(
    "median walk_s",
    statistics.median(r[5] for r in rows),
    "median kend",
    statistics.median(r[0] for r in rows),
    "max walk_s",
    max(r[5] for r in rows),
)
