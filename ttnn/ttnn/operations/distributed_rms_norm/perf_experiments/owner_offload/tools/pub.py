"""Sorted rd_publish end times (us from chip kernel start) of the last N runs, per chip. usage: pub.py csv [N] [zone]"""
import csv, sys
from collections import defaultdict

log = sys.argv[1]
lastn = int(sys.argv[2]) if len(sys.argv) > 2 else 3
zone = sys.argv[3] if len(sys.argv) > 3 else "rd_publish"
with open(log) as fh:
    h = fh.readline()
    mhz = float(h.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
by = defaultdict(list)
for r in rows:
    by[(int(r["run host ID"]) - int(r["PCIe slot"]), int(r["PCIe slot"]))].append(r)
for rid in sorted({k[0] for k in by})[-lastn:]:
    for chip in range(4):
        rr = by[(rid, chip)]
        t0 = min(
            int(r["time[cycles since reset]"])
            for r in rr
            if r["zone name"].endswith("KERNEL") and r["type"] == "ZONE_START"
        )
        e = sorted(
            (int(r["time[cycles since reset]"]) - t0) / mhz
            for r in rr
            if r["zone name"] == zone and r["type"] == "ZONE_END"
        )
        print(rid, chip, " ".join(f"{x:.1f}" for x in e))
