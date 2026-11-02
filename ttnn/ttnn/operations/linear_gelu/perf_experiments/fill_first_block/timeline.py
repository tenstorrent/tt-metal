"""Fill timeline: per zone, start/end relative to the earliest kernel start (us), min / p50 / max over cores.

Usage: python timeline.py _logs/<tag>.csv [run_index (default -1)] [zone,zone,...]
Only each zone's FIRST execution per (core, RISC) is used.
"""
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
idx = int(sys.argv[2]) if len(sys.argv) > 2 else -1
want = sys.argv[3].split(",") if len(sys.argv) > 3 else None
with open(path) as f:
    first = f.readline()
    freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(f, skipinitialspace=True))
run = sorted({int(r["run host ID"]) for r in rows})[idx]
rows = [r for r in rows if int(r["run host ID"]) == run]
t0 = min(int(r["time[cycles since reset]"]) for r in rows if r["zone name"].endswith("-KERNEL"))
starts, spans = {}, defaultdict(list)
seen = set()
for r in rows:
    key = (r["core_x"], r["core_y"], r["RISC processor type"], r["zone name"])
    t = int(r["time[cycles since reset]"]) - t0
    if r["type"] == "ZONE_START" and key not in starts and key not in seen:
        starts[key] = t
    elif r["type"] == "ZONE_END" and key in starts and key not in seen:
        seen.add(key)
        spans[(r["zone name"], r["RISC processor type"])].append((starts.pop(key), t))
us = lambda c: c / freq
for (zone, risc), v in sorted(spans.items(), key=lambda kv: statistics.median(s for s, _ in kv[1])):
    if want and zone not in want:
        continue
    s = sorted(a for a, _ in v)
    e = sorted(b for _, b in v)
    d = sorted(b - a for a, b in v)
    print(
        f"{risc:7s} {zone:20s} n={len(v):3d} start p50={us(statistics.median(s)):6.2f} max={us(s[-1]):6.2f} | "
        f"end min={us(e[0]):6.2f} p50={us(statistics.median(e)):6.2f} max={us(e[-1]):6.2f} | dur p50={us(statistics.median(d)):5.2f} max={us(d[-1]):5.2f}"
    )
