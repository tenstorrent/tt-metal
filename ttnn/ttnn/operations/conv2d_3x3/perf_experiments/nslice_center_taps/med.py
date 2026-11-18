# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Median over cores of phase boundaries per run (no device): python3 med.py <device.csv> [run indices...]"""
import collections
import statistics
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
import zone_report as zr

path = sys.argv[1]
freq, runs = zr.load(path)
keys = sorted(runs)
idxs = [int(x) for x in sys.argv[2:]] or range(len(keys))
Z = [
    ("reader_fill", 1),
    ("reader_center", 0),
    ("reader_center", 1),
    ("compute_untilize", 1),
    ("reader_gather", 0),
    ("reader_gather", 1),
    ("compute_tilize", 0),
    ("compute_tilize", 1),
    ("compute_matmul_walk", 0),
    ("compute_matmul_walk", 1),
]
print(
    "run  " + " ".join(f"{z[0].split('_',1)[1][:7]}{'se'[i]:>1}" for z, i in [(z, z[1]) for z in Z]) + "  kend(med/max)"
)
for idx in idxs:
    ev = runs[keys[idx]]
    t0 = min(e[2] for e in ev)
    per = collections.defaultdict(dict)
    stack = collections.defaultdict(list)
    kend = collections.defaultdict(float)
    for c, risc, t, zone, kind in sorted(ev, key=lambda e: e[2]):
        if kind == "ZONE_START":
            stack[(c, risc, zone)].append(t)
        elif kind == "ZONE_END" and stack[(c, risc, zone)]:
            s = stack[(c, risc, zone)].pop()
            d = per[c]
            if zone not in d:
                d[zone] = [(s - t0) / freq, (t - t0) / freq]
            else:
                d[zone][1] = (t - t0) / freq
            if zone.endswith("-KERNEL"):
                kend[c] = max(kend[c], (t - t0) / freq)
    vals = []
    for z, i in Z:
        xs = [d[z][i] for d in per.values() if z in d]
        vals.append(statistics.median(xs) if xs else float("nan"))
    print(
        f"{idx:3d}  "
        + " ".join(f"{v:8.2f}" for v in vals)
        + f"  {statistics.median(kend.values()):6.2f}/{max(kend.values()):6.2f}"
    )
