"""Per-core zone timeline (us from the earliest kernel start) for selected cores / risc.
usage: r16_timeline.py <profile_log_device.csv> <risc> <zones comma> <cores 'x,y;x,y'>"""
import csv, sys, collections

rows = list(csv.reader(open(sys.argv[1])))[2:]
risc_f, zones = sys.argv[2], sys.argv[3].split(",")
cores = [tuple(int(v) for v in c.split(",")) for c in sys.argv[4].split(";")]
by_run = collections.defaultdict(list)
for r in rows:
    by_run[r[7].strip()].append(r)
run = max(by_run, key=lambda k: len(by_run[k]))
t0 = min(int(r[5]) for r in by_run[run])
open_, ev = {}, collections.defaultdict(list)
for r in by_run[run]:
    core, risc, z, typ, t = (int(r[1]), int(r[2])), r[3].strip(), r[10].strip(), r[11].strip(), int(r[5])
    if risc != risc_f or z not in zones or core not in cores:
        continue
    key = (core, z)
    if typ == "ZONE_START":
        open_.setdefault(key, []).append(t)
    elif typ == "ZONE_END" and open_.get(key):
        s = open_[key].pop()
        ev[core].append(((s - t0) / 1350, (t - t0) / 1350, z))
for c in cores:
    print(c, " ".join(f"{z[5:9]}[{a:.1f}-{b:.1f}]" for a, b, z in sorted(ev[c])))
