"""Kernel span (us) of every run per device in a profile_log_device.csv. usage: python spans.py <csv>"""
import csv, sys
from collections import defaultdict

FREQ = 1350.0
lo, hi = defaultdict(lambda: 1 << 62), defaultdict(int)
with open(sys.argv[1]) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for x in r:
        if x[10].endswith("KERNEL"):
            k = (int(x[0]), int(x[7]))
            t = int(x[5])
            if x[11] == "ZONE_START":
                lo[k] = min(lo[k], t)
            else:
                hi[k] = max(hi[k], t)
for k in sorted(lo):
    print(k, round((hi[k] - lo[k]) / FREQ, 1))
