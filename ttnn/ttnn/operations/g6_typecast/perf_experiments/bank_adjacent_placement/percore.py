# usage: percore.py ORDER [profile_log_device.csv] -> per item, per core (logical x,y): median over launches 2..12 of
#   st  = core's first kernel start - op's first kernel start (launch skew), ns
#   rd  = NCRISC (reader) kernel duration, ns
#   end = core's last kernel end - op's first kernel start, ns
import collections, csv, statistics, sys

CLK = 1.35  # cycles per ns
order = [l.split() for l in open(sys.argv[1]) if l.strip()]
SKIP = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # leading runs to ignore
path = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "/localdev/pmilojevic/campaign/bh-qb-15/slots/chip1/generated/profiler/.logs/profile_log_device.csv"
)
PX = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14]  # profiler x of logical x 0..10 (this chip: phys col 10 harvested)
ev = collections.defaultdict(dict)  # run -> (core, risc, kind) -> t
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for row in r:
        zone = row[10].strip()
        if not zone.endswith("-KERNEL"):
            continue
        px = int(row[1])
        run = int(row[7])
        core = (PX.index(px) if px in PX else 100 + px, int(row[2]) - 2)
        ev[run][(core, row[3].strip(), row[11].strip())] = int(row[5])
runs = sorted(ev)[SKIP:]
assert len(runs) == 12 * len(order), (len(runs), len(order))
for i, (case, mode) in enumerate(order):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    tot = []
    for run in runs[12 * i + 1 : 12 * i + 12]:
        e = ev[run]
        t0 = min(t for (c, rs, k), t in e.items() if k == "ZONE_START")
        t1 = max(t for (c, rs, k), t in e.items() if k == "ZONE_END")
        tot.append((t1 - t0) / CLK)
        cores = {c for (c, rs, k) in e}
        for c in cores:
            st = min(t for (cc, rs, k), t in e.items() if cc == c and k == "ZONE_START")
            en = max(t for (cc, rs, k), t in e.items() if cc == c and k == "ZONE_END")
            per[c]["st"].append((st - t0) / CLK)
            per[c]["end"].append((en - t0) / CLK)
            per[c]["rd"].append((e[(c, "NCRISC", "ZONE_END")] - e[(c, "NCRISC", "ZONE_START")]) / CLK)
            per[c]["wr"].append((e[(c, "BRISC", "ZONE_END")] - e[(c, "BRISC", "ZONE_START")]) / CLK)
    print(f"== {case} {mode} span={statistics.median(tot):.0f}")
    cells = []
    for c in sorted(per, key=lambda c: (c[1], c[0])):
        m = {k: statistics.median(v) for k, v in per[c].items()}
        cells.append(f"{c[0]},{c[1]}: st={m['st']:.0f} rd={m['rd']:.0f} wr={m['wr']:.0f} end={m['end']:.0f}")
    for j in range(0, len(cells), 4):
        print("   " + " | ".join(cells[j : j + 4]))
