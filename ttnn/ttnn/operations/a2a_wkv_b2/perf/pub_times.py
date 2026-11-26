"""Per-block publish times (WPUB zone end, us from program start) over worker cores, last run of a chip."""
import csv, sys
from collections import defaultdict

rows = list(csv.reader(open(sys.argv[2] if len(sys.argv) > 2 else "generated/profiler/.logs/profile_log_device.csv")))[
    2:
]
runs = defaultdict(int)
for r in rows:
    runs[r[0]] = max(runs[r[0]], int(r[7]))
chip = sys.argv[1]
rs = [r for r in rows if r[0] == chip and int(r[7]) == runs[chip]]
st = min(int(r[5]) for r in rs)
ends = defaultdict(list)
for r in rs:
    if r[10].strip() == "WPUB" and r[11].strip() == "ZONE_END":
        ends[(int(r[1]), int(r[2]))].append((int(r[5]) - st) / 1350)
per_block = defaultdict(list)
for c, v in ends.items():
    for i, t in enumerate(sorted(v)):
        per_block[i].append(t)
for i in sorted(per_block):
    v = sorted(per_block[i])
    print(
        f"block {i}: min {v[0]:.1f} p10 {v[len(v)//10]:.1f} med {v[len(v)//2]:.1f} p90 {v[9*len(v)//10]:.1f} max {v[-1]:.1f}"
    )
rows_by_y = defaultdict(list)
for (x, y), v in ends.items():
    rows_by_y[y].append((x, sorted(v)[1]))
for y in sorted(rows_by_y):
    print(y, " ".join(f"{x}:{t:.0f}" for x, t in sorted(rows_by_y[y])))
