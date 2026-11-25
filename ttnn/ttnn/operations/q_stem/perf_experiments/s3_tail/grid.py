"""Logical-grid map of one zone's end time per core for each chip (host only).
    python3 grid.py zones.csv ZONE RISC [start|end]"""
import csv, sys
from collections import defaultdict

path, zone, risc = sys.argv[1:4]
which = sys.argv[4] if len(sys.argv) > 4 else "end"
f = open(path)
f.readline()
R = list(csv.DictReader(f, skipinitialspace=True))
for chip in sorted({r["PCIe slot"] for r in R}, key=int):
    S = [r for r in R if r["PCIe slot"] == chip]
    lid = max(int(r["run host ID"]) for r in S)
    S = [r for r in S if int(r["run host ID"]) == lid]
    xs = sorted({int(r["core_x"]) for r in S})
    ys = sorted({int(r["core_y"]) for r in S})
    t0 = min(
        int(r["time[cycles since reset]"]) for r in S if r["zone name"].endswith("KERNEL") and r["type"] == "ZONE_START"
    )
    v = {}
    for r in S:
        if r["zone name"] == zone and r["RISC processor type"] == risc:
            c = (xs.index(int(r["core_x"])), ys.index(int(r["core_y"])))
            t = (int(r["time[cycles since reset]"]) - t0) / 1350
            if (r["type"] == "ZONE_END") == (which == "end"):
                v[c] = max(v.get(c, t), t) if which == "end" else min(v.get(c, t), t)
    print(f"chip {chip} {zone} {risc} {which}")
    for y in range(len(ys)):
        print("  y%d " % y + " ".join(f"{v[(x, y)]:6.1f}" if (x, y) in v else "     ." for x in range(len(xs))))
