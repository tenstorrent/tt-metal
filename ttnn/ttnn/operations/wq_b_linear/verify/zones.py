"""Custom-zone end times (ns after op start) on a few worker cores, last dispatch. Usage: zones.py [csv]."""
import csv, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
rows = list(csv.reader(open(path)))[2:]
last = max(int(r[7]) for r in rows)
rows = [r for r in rows if int(r[7]) == last]
starts = sorted(int(r[5]) for r in rows if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START")
med = starts[len(starts) // 2]
t0 = min(s for s in starts if abs(s - med) < 1e6)
ev = defaultdict(list)
for r in rows:
    name = r[10].strip()
    if name.endswith("-KERNEL") or name.endswith("-FW"):
        continue
    if abs(int(r[5]) - med) > 1e6:
        continue
    ev[((int(r[1]), int(r[2])), r[3].strip(), name)].append((r[11].strip(), (int(r[5]) - t0) / 1.35))
cores = sorted({k[0] for k in ev})
for c in [c for c in cores if c[0] in (2, 15)] + cores[:: max(1, len(cores) // 3)]:
    for k in sorted(k for k in ev if k[0] == c):
        se = ev[k]
        ends = [round(t) for typ, t in se if typ == "ZONE_END"]
        sts = [round(t) for typ, t in se if typ == "ZONE_START"]
        print(c, k[1], k[2], "start", sts, "end", ends)
