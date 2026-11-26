"""Aggregate delivered DRAM bytes per 4 us bucket (chip 0, last replay) from reader_attn_block zone ends."""
import csv, sys, bisect
from collections import defaultdict

FREQ = 1350.0
rows = list(csv.reader(open(sys.argv[1])))[2:]
chip_sel = sys.argv[2] if len(sys.argv) > 2 else "0"
kst = []
for r in rows:
    if r[0] == chip_sel and r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START":
        kst.append(int(r[5]))
kst.sort()
starts = [kst[0]]
for a, b in zip(kst, kst[1:]):
    if b - a > 5 * FREQ:
        starts.append(b)
t0 = starts[-1]
ends = defaultdict(list)
for r in rows:
    if r[0] != chip_sel or r[3].strip() != "NCRISC":
        continue
    t = int(r[5])
    if t < t0:
        continue
    if r[10].strip() == "reader_attn_block" and r[11].strip() == "ZONE_END":
        ends[(r[1], r[2])].append((t - t0) / FREQ)
buckets = defaultdict(float)
for core, ts in ends.items():
    ts.sort()
    for i, t in enumerate(ts):
        buckets[int(t // 4)] += 32768 + (69632 if i == 0 else 0)
tot = 0
for b in sorted(buckets):
    tot += buckets[b]
    print(f"{4*b:3d}-{4*b+4:3d} us: {buckets[b]/4e3:6.0f} GB/s  cum {tot/1e6:5.2f} MB")
