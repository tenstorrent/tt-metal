"""Per-role end times (us from program start) from generated/profiler/.logs/profile_log_device.csv.

Usage: python3 parse_profile.py [csv] — prints, per run (host id) and chip, the max end time of
worker NCRISC/TRISC/BRISC and of the fabric_tx cores (cores without a TRISC kernel), per tx core.
"""
import csv, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
FREQ = 1350.0
start = defaultdict(lambda: 1 << 62)
end = defaultdict(dict)  # (chip, run) -> {(x,y,risc): end}
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for row in r:
        chip, x, y, risc, _, t, _, run, *_rest = row
        zone, typ = row[10].strip(), row[11].strip()
        if not zone.endswith("-KERNEL"):
            continue
        key = (chip, int(run))
        t = int(t)
        if typ == "ZONE_START":
            start[key] = min(start[key], t)
        elif typ == "ZONE_END":
            k = (int(x), int(y), risc.strip())
            end[key][k] = max(end[key].get(k, 0), t)
runs = sorted({k[1] for k in end})
for run in runs[-8:]:
    print(f"== run {run}")
    for key in sorted(k for k in end if k[1] == run):
        s = start[key]
        e = end[key]
        trisc_cores = {(x, y) for (x, y, rk) in e if rk.startswith("TRISC")}

        def us(v):
            return (v - s) / FREQ

        w = defaultdict(float)
        tx = defaultdict(dict)
        for (x, y, rk), v in e.items():
            if (x, y) in trisc_cores:
                role = "TRISC" if rk.startswith("TRISC") else rk
                w[role] = max(w[role], us(v))
            else:
                tx[(x, y)][rk] = us(v)
        txs = " ".join(
            f"({x},{y}):NC{d.get('NCRISC',0):.1f}/BR{d.get('BRISC',0):.1f}" for (x, y), d in sorted(tx.items())
        )
        print(f"chip {key[0]}: worker NC {w['NCRISC']:.1f} TR {w['TRISC']:.1f} BR {w['BRISC']:.1f} | tx {txs}")
