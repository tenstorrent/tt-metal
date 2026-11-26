#!/usr/bin/env python3
"""Absolute zone [start, end] (us from the run's earliest marker) per core: zabs.py <device.csv> <run id> <risc> <zone>
[s|e|d] (prints a logical grid of start / end / duration of the zone's first occurrence)."""
import csv, sys

path, run, risc, zone = sys.argv[1:5]
what = sys.argv[5] if len(sys.argv) > 5 else "e"
with open(path) as f:
    head = f.readline()
    mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = [r for r in csv.DictReader(f, skipinitialspace=True) if r["run host ID"] == str(run)]
t0 = min(int(r["time[cycles since reset]"]) for r in rows)
se = {}
for r in rows:
    if r["RISC processor type"] != risc or r["zone name"] != zone:
        continue
    c = (int(r["core_x"]), int(r["core_y"]))
    t = (int(r["time[cycles since reset]"]) - t0) / mhz
    d = se.setdefault(c, [None, None])
    if r["type"] == "ZONE_START" and d[0] is None:
        d[0] = t
    elif r["type"] == "ZONE_END" and d[1] is None:
        d[1] = t
xs = sorted({c[0] for c in se})
ys = sorted({c[1] for c in se})
val = lambda d: d[0] if what == "s" else d[1] if what == "e" else d[1] - d[0]
vs = [val(d) for d in se.values()]
print(f"{risc} {zone} {what}: max {max(vs):.1f} min {min(vs):.1f}")
for y in ys:
    print(f"y{ys.index(y):2d} " + " ".join(f"{val(se[(x, y)]):7.1f}" if (x, y) in se else "      -" for x in xs))
