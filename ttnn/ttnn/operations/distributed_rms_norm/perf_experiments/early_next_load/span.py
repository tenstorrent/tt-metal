"""Trace wall time per shape: per device, (FW END of traced call 6 - FW START of traced call 1) in us (one clock per
device); reports the max over devices (= the 6-call trace's wall time, launch offset aside) / 6 = per-call throughput.
usage: span.py [tags...]"""
import csv, sys, collections, glob, os

E = os.path.dirname(os.path.abspath(__file__))
tags = sys.argv[1:] or sorted(os.path.basename(p)[4:-4] for p in glob.glob(f"{E}/logs/csv_*.txt"))
for tag in tags:
    path = open(f"{E}/logs/csv_{tag}.txt").read().strip()
    rows = [r for r in csv.DictReader(open(path)) if "GenericOp" in r["OP CODE"] and r["DEVICE KERNEL DURATION [ns]"]]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["DEVICE ID"]].append((int(r["DEVICE FW START CYCLE"]), int(r["DEVICE FW END CYCLE"])))
    out = []
    for s in range(6):
        sp = [(by[d][s * 7 + 6][1] - by[d][s * 7 + 1][0]) / 1350.0 for d in sorted(by)]  # BH 1350 MHz
        out.append(f"{max(sp) / 6:6.2f}")
    print(f"{tag:14s}", " ".join(out))
