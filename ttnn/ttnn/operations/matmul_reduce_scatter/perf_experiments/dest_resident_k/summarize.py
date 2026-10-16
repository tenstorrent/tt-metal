"""Per-device DEVICE KERNEL DURATION [ns] of the generic-op rows in an ops_perf_results CSV (last 3 per device =
the traced replays); op latency = max over devices per replay."""
import csv
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
per_dev = defaultdict(list)
for r in rows:
    if "generic" not in r.get("OP CODE", "").lower():
        continue
    if not r["DEVICE KERNEL DURATION [ns]"].strip():
        continue
    per_dev[int(r["DEVICE ID"])].append(float(r["DEVICE KERNEL DURATION [ns]"]))
n = min(len(v) for v in per_dev.values())
k = min(3, n)
for d in sorted(per_dev):
    print(f"dev {d}: " + " ".join(f"{x/1000:.1f}" for x in per_dev[d][-k:]))
lat = [max(per_dev[d][-k + i] for d in per_dev) for i in range(k)]
print("op latency (max over devices) per replay us:", " ".join(f"{x/1000:.2f}" for x in lat))
