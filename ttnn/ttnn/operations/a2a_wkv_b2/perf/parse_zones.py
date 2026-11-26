"""Sum of TX zone durations (us) per tx core, last run per chip: python3 parse_zones.py [csv]."""
import csv, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
open_t, tot, cnt, last = {}, defaultdict(float), defaultdict(int), {}
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    rows = list(r)
runs = defaultdict(int)
for row in rows:
    runs[row[0]] = max(runs[row[0]], int(row[7]))
for row in rows:
    chip, x, y, risc, _, t, _, run = row[:8]
    zone, typ = row[10].strip(), row[11].strip()
    if not zone.startswith("TX") or int(run) != runs[chip]:
        continue
    k = (chip, x, y, zone)
    if typ == "ZONE_START":
        open_t[k] = int(t)
    elif typ == "ZONE_END" and k in open_t:
        tot[k] += (int(t) - open_t.pop(k)) / 1350.0
        cnt[k] += 1
for k in sorted(tot):
    print(f"chip {k[0]} ({k[1]},{k[2]}) {k[3]:10s} n={cnt[k]:4d} total {tot[k]:7.1f} us")
