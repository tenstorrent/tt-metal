"""Per-occurrence timeline of chosen zones: median over cores of the i-th occurrence's start / end (us from
the earliest kernel start). python occ.py <device.csv> <risc> <zone> [<zone> ...]"""
import collections, csv, statistics, sys

path, risc_sel, zones = sys.argv[1], sys.argv[2], set(sys.argv[3:])
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
t0 = None
open_, occ = {}, collections.defaultdict(list)
for r in rows:
    if len(r) < 12:
        continue
    core, risc, t, zone, typ = (r[1].strip(), r[2].strip()), r[3].strip(), int(r[5]), r[10].strip(), r[11].strip()
    if zone.endswith("-KERNEL") and typ == "ZONE_START":
        t0 = t if t0 is None else min(t0, t)
    if risc != risc_sel or zone not in zones:
        continue
    key = (core, zone)
    if typ == "ZONE_START":
        open_[key] = t
    elif key in open_:
        occ[key].append((open_.pop(key), t))
for zone in sorted(zones):
    per = [v for (c, z), v in occ.items() if z == zone]
    n = max(len(v) for v in per) if per else 0
    for i in range(n):
        vals = [v[i] for v in per if len(v) > i]
        s = statistics.median((a - t0) / freq for a, _ in vals)
        e = statistics.median((b - t0) / freq for _, b in vals)
        d = statistics.median((b - a) / freq for a, b in vals)
        print(
            f"{zone:20} #{i} start {s:6.2f} end {e:6.2f} dur_med {d:5.2f} dur_max {max((b-a)/freq for a,b in vals):5.2f}"
        )
