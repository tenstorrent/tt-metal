"""Kernel duration + per-zone summary of one copied profile log (last run host ID).

Usage: python summarize.py _logs/<tag>.csv variant1,variant2,... [tag]
Run host IDs in increasing order map to the variants in order (one op per variant).
Kernel duration = max(*-KERNEL end) - min(*-KERNEL start) over all cores (same span the ops CSV reports).
"""
import csv
import statistics
import sys
from collections import defaultdict


def summarize(label, rows, freq):
    k_start, k_end = None, None
    open_ = {}
    tot = defaultdict(lambda: defaultdict(int))
    for r in rows:
        t = int(r["time[cycles since reset]"])
        zone = r["zone name"]
        if zone.endswith("-KERNEL"):
            if r["type"] == "ZONE_START":
                k_start = t if k_start is None else min(k_start, t)
            else:
                k_end = t if k_end is None else max(k_end, t)
        key = ((r["core_x"], r["core_y"]), r["RISC processor type"], zone)
        if r["type"] == "ZONE_START":
            open_[key] = t
        elif r["type"] == "ZONE_END" and key in open_:
            tot[(zone, r["RISC processor type"])][key[0]] += t - open_.pop(key)
    print(f"{label}: kernel {(k_end - k_start) / freq * 1000:.0f} ns")
    for (zone, risc), per in sorted(tot.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if zone.endswith("-FW"):
            continue
        vals = list(per.values())
        print(f"  {risc:7s} {zone:22s} max={max(vals) / freq:7.2f} p50={statistics.median(vals) / freq:7.2f} us")


path, names = sys.argv[1], sys.argv[2].split(",")
tag = sys.argv[3] if len(sys.argv) > 3 else ""
with open(path) as f:
    first = f.readline()
    freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    all_rows = list(csv.DictReader(f, skipinitialspace=True))
by_run = defaultdict(list)
for r in all_rows:
    by_run[int(r["run host ID"])].append(r)
runs = sorted(by_run)
if len(runs) != len(names):
    print(f"WARNING: {len(runs)} runs in log vs {len(names)} variants; mapping the last {len(names)}")
for name, run in zip(names, runs[-len(names) :]):
    summarize(f"{tag} {name}", by_run[run], freq)
