"""Per-core zone timeline (us from the op's earliest start) for one RISC.
usage: r19_timeline.py <profile_log_device.csv> <risc> [zones comma] [max cores]"""
import csv, sys, collections

rows = list(csv.reader(open(sys.argv[1])))[2:]
risc_f = sys.argv[2]
zones_f = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else None
maxc = int(sys.argv[4]) if len(sys.argv) > 4 else 200
by_run = collections.defaultdict(list)
for r in rows:
    by_run[r[7].strip()].append(r)
run = max(by_run, key=lambda k: len(by_run[k]))
R = by_run[run]
t0 = min(int(r[5]) for r in R)
open_ = {}
ev = collections.defaultdict(list)
for r in R:
    core, risc, z, typ, t = (int(r[1]), int(r[2])), r[3].strip(), r[10].strip(), r[11].strip(), int(r[5]) - t0
    if risc != risc_f or (zones_f and z not in zones_f):
        continue
    if typ == "ZONE_START":
        open_.setdefault((core, z), []).append(t)
    elif typ == "ZONE_END" and open_.get((core, z)):
        s = open_[(core, z)].pop()
        ev[core].append((s / 1350, t / 1350, z))
for core in sorted(ev, key=lambda c: max(e[1] for e in ev[c]))[:maxc]:
    print(core, " ".join(f"{z}[{s:.1f}-{e:.1f}]" for s, e, z in sorted(ev[core])))
