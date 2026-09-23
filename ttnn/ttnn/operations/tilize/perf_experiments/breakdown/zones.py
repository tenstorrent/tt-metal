"""Summarize a --profile run: per-op DEVICE KERNEL DURATION and per-zone per-RISC stats.

usage: python zones.py [report_dir]   (default: newest generated/profiler/reports/*)
Per zone and RISC-V: cores that recorded it, executions per core, and the per-core SUM of the
zone's durations (cycles; WH AICLK here is 1000 MHz so cycles == ns), as p50 / max across cores.
Also the per-RISC-V *-KERNEL span (p50 / max) and the marker count per (core, RISC) max.
Only the LAST program's markers are used when several ops ran (grouped by run host ID).
"""
import csv, glob, os, statistics, sys
from collections import defaultdict

root = os.path.join(os.path.dirname(__file__), "../../../../../..")
rep = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(root, "generated/profiler/reports/*/")))[-1]
ops = glob.glob(os.path.join(rep, "ops_perf_results*.csv"))[0]
with open(ops) as f:
    rows = list(csv.DictReader(f))
for r in rows:
    print(
        f"OP {r['OP CODE']} host_id={r['GLOBAL CALL COUNT']} cores={r['CORE COUNT']} "
        f"DEVICE KERNEL DURATION [ns]={r['DEVICE KERNEL DURATION [ns]']}"
    )
dev = os.path.join(rep, "profile_log_device.csv")
with open(dev) as f:
    next(f)
    rd = csv.reader(f)
    hdr = [h.strip() for h in next(rd)]
    data = [dict(zip(hdr, [c.strip() for c in row])) for row in rd]
ids = sorted({d["run host ID"] for d in data}, key=int)
want = sys.argv[2] if len(sys.argv) > 2 else ids[-1]
data = [d for d in data if d["run host ID"] == want]
open_ = {}
per = defaultdict(lambda: defaultdict(list))  # (zone, risc) -> core -> [durations]
markers = defaultdict(int)
for d in data:
    key = (d["core_x"], d["core_y"], d["RISC processor type"])
    markers[key] += 1
    zk = key + (d["zone name"],)
    t = int(d["time[cycles since reset]"])
    if d["type"] == "ZONE_START":
        open_.setdefault(zk, []).append(t)
    elif d["type"] == "ZONE_END" and open_.get(zk):
        s = open_[zk].pop()
        per[(d["zone name"], d["RISC processor type"])][(d["core_x"], d["core_y"])].append(t - s)
print(f"run host ID {want}; max markers per (core,RISC) = {max(markers.values())}")
order = sorted(per, key=lambda k: (k[1], k[0]))
print(f"{'zone':32s} {'risc':8s} {'cores':>5s} {'n/core':>6s} {'sum p50':>9s} {'sum max':>9s} {'each p50':>8s}")
for k in order:
    cores = per[k]
    sums = [sum(v) for v in cores.values()]
    each = [x for v in cores.values() for x in v]
    n = statistics.median([len(v) for v in cores.values()])
    print(
        f"{k[0]:32s} {k[1]:8s} {len(cores):5d} {n:6.0f} {statistics.median(sums):9.0f} {max(sums):9.0f} {statistics.median(each):8.0f}"
    )
