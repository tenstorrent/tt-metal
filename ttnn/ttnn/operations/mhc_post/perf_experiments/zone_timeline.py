"""Perf 1 breakdown tool: per-zone timeline of the last mhc_post run per device in a profile_log_device.csv.

usage: python zone_timeline.py <profile_log_device.csv> [device]
Times in us relative to the earliest kernel start on that device (1350 MHz). For each zone: executions, cores,
start (min / median / max), duration (median / max), end (median / max)."""
import csv, sys, statistics as st
from collections import defaultdict

FREQ = 1350.0
path = sys.argv[1]
want = int(sys.argv[2]) if len(sys.argv) > 2 else None
rows = []
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for x in r:
        rows.append(x)
by_dev = defaultdict(list)
for x in rows:
    by_dev[int(x[0])].append(x)
for dev, xs in sorted(by_dev.items()):
    if want is not None and dev != want:
        continue
    # the last run host id that has user zones or worker kernels on many cores
    runs = sorted({int(x[7]) for x in xs})
    run = max(r for r in runs if sum(1 for x in xs if int(x[7]) == r and x[10].endswith("KERNEL")) > 200)
    xs = [x for x in xs if int(x[7]) == run]
    t0 = min(int(x[5]) for x in xs if x[10].endswith("KERNEL") and x[11] == "ZONE_START")
    t1 = max(int(x[5]) for x in xs if x[10].endswith("KERNEL") and x[11] == "ZONE_END")
    open_ = {}
    z = defaultdict(list)  # name -> [(core, risc, start, end)]
    for x in sorted(xs, key=lambda x: int(x[5])):
        key = (x[1], x[2], x[3], x[10])
        t = (int(x[5]) - t0) / FREQ
        if x[11] == "ZONE_START":
            open_.setdefault(key, []).append(t)
        elif x[11] == "ZONE_END" and open_.get(key):
            s = open_[key].pop()
            z[(x[10], x[3] if "KERNEL" in x[10] or "FW" in x[10] else x[3])].append(((x[1], x[2]), s, t))
    print(f"== device {dev} run {run}: kernel span {(t1 - t0) / FREQ:.1f} us")
    print(
        f"{'zone':24s}{'risc':8s}{'n':>6s}{'cores':>6s}  {'start min/med/max':>20s}  {'dur med/max':>14s}  {'end med/max':>14s}  {'sum/core med':>12s}"
    )
    order = sorted(z.items(), key=lambda kv: st.median([s for _, s, _ in kv[1]]))
    for (name, risc), ev in order:
        if name.endswith("-FW"):
            continue
        cores = {c for c, _, _ in ev}
        starts = [s for _, s, _ in ev]
        durs = [e - s for _, s, e in ev]
        ends = [e for _, _, e in ev]
        per_core = defaultdict(float)
        for c, s, e in ev:
            per_core[c] += e - s
        print(
            f"{name:24s}{risc:8s}{len(ev):6d}{len(cores):6d}  {min(starts):6.1f}/{st.median(starts):6.1f}/{max(starts):6.1f}"
            f"  {st.median(durs):6.2f}/{max(durs):6.2f}  {st.median(ends):6.1f}/{max(ends):6.1f}  {st.median(per_core.values()):10.2f}"
        )
