"""Per device, per traced call DEVICE KERNEL DURATION (us) for shape block s of a test_perf_ring ops CSV.
usage: per_device.py <csv> [shape_idx=0] [n_calls=6]"""
import csv, sys, collections

path = sys.argv[1]
s = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
rows = [r for r in csv.DictReader(open(path)) if "GenericOp" in r["OP CODE"] and r["DEVICE KERNEL DURATION [ns]"]]
by = collections.defaultdict(list)
for r in rows:
    by[r["DEVICE ID"]].append(int(r["DEVICE KERNEL DURATION [ns]"]))
per = 1 + n
for d in sorted(by):
    print(d, [round(v / 1000, 2) for v in by[d][s * per + 1 : s * per + per]])
