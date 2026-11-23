"""Zone start/end (ns after op start) of the x cores (x_read_*, x_help, x_mcast_*), last dispatch. Usage: timeline.py csv"""
import csv, sys
from collections import defaultdict

rows = list(csv.reader(open(sys.argv[1])))[2:]
run = sorted({int(r[7]) for r in rows})[-1]
rr = [r for r in rows if int(r[7]) == run]
ks = [int(r[5]) for r in rr if r[10].strip().endswith("-KERNEL")]
med = sorted(ks)[len(ks) // 2]
t0 = min(t for t in ks if abs(t - med) < 1e6)
cores = {(r[1], r[2]) for r in rr if r[10].strip().startswith(("x_read", "x_help", "x_mcast_issue"))}
agg = defaultdict(list)
for r in rr:
    if (r[1], r[2]) in cores and abs(int(r[5]) - med) < 1e6:
        agg[(r[1].strip(), r[2].strip(), r[3].strip(), r[10].strip(), r[11].strip())].append(
            round((int(r[5]) - t0) / 1.35)
        )
for k, v in sorted(agg.items(), key=lambda kv: kv[1][0]):
    print(*k, v[0], v[-1], len(v))
