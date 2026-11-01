"""Per-core end (us from op start) of the first instance of a zone, per run: min/p50/max + by core y.

usage: zone_ends.py <zone> [csv]
"""
import csv
import statistics
import sys
from collections import defaultdict

zone = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 else "last_profile_log_device.csv"
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
t0 = defaultdict(lambda: 1 << 62)
first_end = defaultdict(dict)
for r in rows[1:]:
    run = r[idx["run host ID"]].strip()
    core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
    t = int(r[idx["time[cycles since reset]"]])
    t0[run] = min(t0[run], t)
    if r[idx["zone name"]].strip() == zone and r[idx["type"]].strip() == "ZONE_END":
        first_end[run].setdefault(core, t)
for run, ends in first_end.items():
    us = {c: (t - t0[run]) / freq for c, t in ends.items()}
    v = sorted(us.values())
    by_y = defaultdict(list)
    for c, x in us.items():
        by_y[c[1]].append(x)
    ys = " ".join(f"y{y}:{statistics.mean(x):.1f}" for y, x in sorted(by_y.items()))
    print(f"run {run} {zone} n={len(v)} min {v[0]:.1f} p50 {v[len(v)//2]:.1f} max {v[-1]:.1f} | {ys}")
