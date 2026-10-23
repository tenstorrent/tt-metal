"""All logs/csv_<tag>.txt runs: per shape, slowest-device median (us) and fastest-device median (the last-launched chip,
which waits on nobody: closest to the kernel's own critical path), per tag. usage: summarize.py [tags...]"""
import csv, sys, statistics, collections, glob, os

E = os.path.dirname(os.path.abspath(__file__))
tags = sys.argv[1:] or sorted(os.path.basename(p)[4:-4] for p in glob.glob(f"{E}/logs/csv_*.txt"))
names = ["640x1792", "32x1792", "2048x1280", "4096x1024", "2x4x128x512", "1024x2048"]
for tag in tags:
    path = open(f"{E}/logs/csv_{tag}.txt").read().strip()
    rows = [r for r in csv.DictReader(open(path)) if "GenericOp" in r["OP CODE"] and r["DEVICE KERNEL DURATION [ns]"]]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["DEVICE ID"]].append(int(r["DEVICE KERNEL DURATION [ns]"]))
    devs = sorted(by)
    out = []
    for s in range(6):
        calls = range(s * 7 + 1, s * 7 + 7)
        mx = statistics.median(max(by[d][i] for d in devs) for i in calls) / 1000
        mn = statistics.median(min(by[d][i] for d in devs) for i in calls) / 1000
        out.append(f"{mx:6.2f}/{mn:5.2f}")
    print(f"{tag:14s}", " ".join(out))
print("columns (slowest/fastest device median, us):", " ".join(names))
