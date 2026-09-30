"""Per-zone summary of profile_log_device.csv: per run host ID, per (RISC, zone): per-core summed ns
(max / p50 / min across cores), executions per core, and marker counts per (core, RISC)."""
import csv, sys, statistics
from collections import defaultdict

path = sys.argv[1]
MHZ = 1350.0
starts = {}
acc = defaultdict(lambda: defaultdict(float))  # (run, risc, zone) -> core -> cycles
cnt = defaultdict(lambda: defaultdict(int))
markers = defaultdict(int)
with open(path) as f:
    next(f)
    r = csv.reader(f)
    hdr = [h.strip() for h in next(r)]
    for row in r:
        row = [x.strip() for x in row]
        core = (row[1], row[2])
        risc = row[3]
        t = int(row[5])
        run = row[7]
        zone = row[10]
        typ = row[11]
        key = (run, risc, zone, core)
        markers[(run, core, risc)] += 1
        if typ == "ZONE_START":
            starts.setdefault(key, []).append(t)
        elif typ == "ZONE_END" and starts.get(key):
            t0 = starts[key].pop()
            acc[(run, risc, zone)][core] += t - t0
            cnt[(run, risc, zone)][core] += 1
for run in sorted({k[0] for k in acc}):
    print(f"=== run {run}")
    for (rn, risc, zone), per in sorted(acc.items()):
        if rn != run or zone.endswith("-FW"):
            continue
        v = sorted(per.values())
        c = list(cnt[(rn, risc, zone)].values())
        print(
            f"  {risc:7s} {zone:22s} cores {len(v):3d}  max {v[-1]/MHZ:9.1f}us  p50 {statistics.median(v)/MHZ:9.1f}us  min {v[0]/MHZ:9.1f}us  exec/core {max(c)}"
        )
    mk = [n for (rn, _, _), n in markers.items() if rn == run]
    print(f"  max markers per (core,RISC): {max(mk)}")
