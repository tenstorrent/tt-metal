"""Back-to-back trace replays (one profiler run): per chip, each replay's device span (first kernel start ->
last kernel end, us) and the idle gap before it (previous replay's last kernel end -> this start).
Replays are split where kernel starts jump by more than 5 us. Usage: python3 trace_gaps.py [csv]"""
import csv, sys
from collections import defaultdict

rows = list(csv.reader(open(sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv")))[
    2:
]
zones = defaultdict(list)  # chip -> [(start, end)] per kernel zone
open_ = {}
for r in rows:
    if not r[10].strip().endswith("-KERNEL"):
        continue
    key = (r[0], r[1], r[2], r[3])
    t = int(r[5])
    if r[11].strip() == "ZONE_START":
        open_[key] = t
    elif key in open_:
        zones[int(r[0])].append((open_.pop(key), t))
for chip in sorted(zones):
    z = sorted(zones[chip])
    groups, cur = [], [z[0]]
    for a in z[1:]:
        if a[0] - cur[-1][0] > 5 * 1350:
            groups.append(cur)
            cur = []
        cur.append(a)
    groups.append(cur)
    out, prev_end = [], None
    for g in groups:
        s, e = min(a for a, _ in g), max(b for _, b in g)
        out.append(f"{(e - s) / 1350:.1f}" + (f"(+{(s - prev_end) / 1350:.1f})" if prev_end else ""))
        prev_end = e
    print(f"chip {chip}: " + " ".join(out))
