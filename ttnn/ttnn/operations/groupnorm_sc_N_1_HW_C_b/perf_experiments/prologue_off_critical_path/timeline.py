#!/usr/bin/env python3
"""Per-run timeline of selected zones on one core (start/end in ns relative to the earliest kernel start of the run).
usage: timeline.py <profile_log_device.csv> <core_x> <core_y> [zone,zone,...]"""
import csv, sys
from collections import defaultdict

p, cx, cy = sys.argv[1], sys.argv[2], sys.argv[3]
want = sys.argv[4].split(",") if len(sys.argv) > 4 else None
f = open(p)
head = f.readline()
mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
k = 1000 / mhz
rows = list(csv.DictReader(f, skipinitialspace=True))
t0 = {}
for r in rows:
    if r["type"] == "ZONE_START" and r["zone name"].endswith("-KERNEL"):
        run = r["run host ID"]
        t = int(r["time[cycles since reset]"])
        t0[run] = min(t0.get(run, t), t)
ev = defaultdict(list)
st = defaultdict(list)
for r in rows:
    if r["type"] not in ("ZONE_START", "ZONE_END") or r["core_x"] != cx or r["core_y"] != cy:
        continue
    run, risc, n, t = r["run host ID"], r["RISC processor type"], r["zone name"], int(r["time[cycles since reset]"])
    if want and n not in want and not n.endswith("-KERNEL"):
        continue
    key = (run, risc, n)
    if r["type"] == "ZONE_START":
        st[key].append(t)
    elif st[key]:
        s = st[key].pop()
        ev[run].append(((s - t0[run]) * k, (t - t0[run]) * k, risc, n))
for run in sorted(ev, key=int):
    print(f"--- run {run}")
    for s, e, risc, n in sorted(ev[run]):
        print(f"  {s:9.0f} {e:9.0f} {e-s:8.0f}  {risc:7s} {n}")
