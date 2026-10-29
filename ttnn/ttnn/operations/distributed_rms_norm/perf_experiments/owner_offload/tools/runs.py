"""Per run, per chip: kernel duration and max zone end (us, relative to the chip's earliest kernel start) of a zone
list. usage: runs.py <csv> [zones] [last_n_runs]"""
import csv, sys
from collections import defaultdict

log = sys.argv[1]
zones = (
    sys.argv[2]
    if len(sys.argv) > 2 and sys.argv[2]
    else "rd_land,rd_publish,prt_exchange,rd_collect,cmp_fold,rd_allgather,rd_inv_col0,wr_store"
).split(",")
lastn = int(sys.argv[3]) if len(sys.argv) > 3 else 9
with open(log) as fh:
    h = fh.readline()
    mhz = float(h.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
by = defaultdict(list)
for r in rows:
    by[(int(r["run host ID"]) - int(r["PCIe slot"]), int(r["PCIe slot"]))].append(r)
rids = sorted({k[0] for k in by})[-lastn:]
print("run chip  dur   " + " ".join(f"{z[:11]:>11s}" for z in zones))
for rid in rids:
    for chip in range(4):
        rr = by.get((rid, chip), [])
        if not rr:
            continue
        st = defaultdict(list)
        ends = defaultdict(int)
        t0 = None
        tend = 0
        for r in rr:
            t = int(r["time[cycles since reset]"])
            z = r["zone name"]
            k = (r["core_x"], r["core_y"], r["RISC processor type"], z)
            if z.endswith("KERNEL"):
                if r["type"] == "ZONE_START":
                    t0 = t if t0 is None else min(t0, t)
                else:
                    tend = max(tend, t)
            if r["type"] == "ZONE_END":
                ends[z] = max(ends[z], t)
        ns = lambda c: (c - t0) / mhz
        print(f"{rid:5d} {chip}  {ns(tend):5.2f} " + " ".join(f"{ns(ends[z]) if ends[z] else 0:11.2f}" for z in zones))
