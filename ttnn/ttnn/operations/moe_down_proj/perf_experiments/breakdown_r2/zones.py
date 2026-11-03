"""Summarize device zones from profile_log_device.csv (last run id per core).

Usage: python zones.py [profile_log_device.csv]
Per (zone, RISC): cores, executions/core, per-core total cycles (max / p50), and kernel span coverage.
"""
import csv
import statistics
import sys
from collections import defaultdict

path = (
    sys.argv[1] if len(sys.argv) > 1 else "/localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv"
)
with open(path) as f:
    first = f.readline()
    freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(f, skipinitialspace=True))
run_id = max(int(r["run host ID"]) for r in rows)
open_ = {}
tot = defaultdict(lambda: defaultdict(int))
cnt = defaultdict(lambda: defaultdict(int))
markers = defaultdict(int)
last_end = defaultdict(int)
for r in rows:
    if int(r["run host ID"]) != run_id:
        continue
    core = (r["core_x"], r["core_y"])
    risc = r["RISC processor type"]
    key = (core, risc, r["zone name"])
    t = int(r["time[cycles since reset]"])
    markers[(core, risc)] += 1
    if r["type"] == "ZONE_START":
        open_[key] = t
    elif r["type"] == "ZONE_END" and key in open_:
        d = t - open_.pop(key)
        tot[(r["zone name"], risc)][core] += d
        cnt[(r["zone name"], risc)][core] += 1
        last_end[(core, risc, r["zone name"])] = t
print(f"run {run_id}, {freq} MHz; per-core totals in us (max / p50), exec per core (max)")
for (zone, risc), per in sorted(tot.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    vals = list(per.values())
    us = lambda c: c / freq
    print(
        f"{risc:7s} {zone:28s} cores={len(vals):3d} exec={max(cnt[(zone,risc)].values()):3d} "
        f"max={us(max(vals)):8.2f} p50={us(statistics.median(vals)):8.2f}"
    )
print("max markers per (core,RISC):", max(markers.values()))
