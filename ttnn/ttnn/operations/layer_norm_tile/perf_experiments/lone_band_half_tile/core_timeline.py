"""Per-core end times and the zone timeline of the latest-ending cores (host-only CSV parsing).
usage: core_timeline.py [profile_log_device.csv] [n_slowest]"""
import csv
import sys
from collections import defaultdict

path = (
    sys.argv[1]
    if len(sys.argv) > 1 and sys.argv[1] != "-"
    else "/localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv"
)
nslow = int(sys.argv[2]) if len(sys.argv) > 2 else 2
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
runs = defaultdict(lambda: defaultdict(list))
open_ = {}
for r in rows[1:]:
    r = [c.strip() for c in r]
    run = r[idx["run host ID"]]
    core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
    k = (run, core, r[idx["RISC processor type"]], r[idx["zone name"]])
    t = int(r[idx["time[cycles since reset]"]])
    if r[idx["type"]] == "ZONE_START":
        open_[k] = t
    elif k in open_:
        runs[run][core].append((k[2], k[3], open_.pop(k), t))
for run, ev in runs.items():
    if not any(z.startswith("compute_") for c in ev for (_, z, _, _) in ev[c]):
        continue
    t0 = min(s for c in ev for (_, z, s, _) in ev[c] if z.endswith("KERNEL"))
    us = lambda t: (t - t0) / freq
    end = {c: max(e[3] for e in ev[c] if e[1].endswith("KERNEL")) for c in ev}
    ncomp = lambda c: sum(
        1 for e in ev[c] if e[0] == "TRISC_1" and e[1] in ("compute_add_and_stats", "compute_lone_add_and_stats")
    )
    lone = lambda c: any(e[1].startswith("compute_lone") for e in ev[c])
    order = sorted(ev, key=lambda c: end[c])
    print(f"run {run}: op span {us(max(end.values())):.2f} us, cores {len(ev)}")
    byrows = defaultdict(list)
    for c in ev:
        byrows[(ncomp(c), lone(c))].append(us(end[c]))
    for k, v in sorted(byrows.items()):
        v.sort()
        print(f"  rows={k[0]} lone_zone={k[1]}: n={len(v)} end min {v[0]:.1f} med {v[len(v)//2]:.1f} max {v[-1]:.1f}")
    for c in order[-nslow:]:
        print(f"  core {c} end {us(end[c]):.2f}")
        for risc, z, s, e in sorted(ev[c], key=lambda e: e[2]):
            if risc == "TRISC_1" or risc in ("BRISC", "NCRISC"):
                print(f"    {risc:7s} {z:34s} {us(s):6.2f} -> {us(e):6.2f} ({(e-s)/freq:5.2f})")
