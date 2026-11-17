"""Summarize genify_matmul device zones (host-side CSV parsing only; no device code).
Usage: python3 zones.py [profile_log_device.csv]
Times in us relative to the earliest kernel start. Per zone: count/core, p50/max of per-core sum,
and p50/max of per-core first-start / last-end."""
import csv, sys, statistics as st
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
f = open(path)
hdr = f.readline()
freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rows = list(csv.reader(f))[1:]
open_ = {}
z = defaultdict(lambda: defaultdict(list))  # zone -> core -> [(s,e)]
for r in rows:
    core = (r[1], r[2], r[3])
    name = r[10].strip()
    typ = r[11].strip()
    t = int(r[5])
    key = (core, name)
    if typ == "ZONE_START":
        open_.setdefault(key, []).append(t)
    elif typ == "ZONE_END" and open_.get(key):
        z[name][core].append((open_[key].pop(), t))
t0 = min(s for c in z["BRISC-KERNEL"].values() for s, _ in c)
us = lambda c: c / freq
for name in [
    "BRISC-KERNEL",
    "NCRISC-KERNEL",
    "TRISC-KERNEL",
    "cmp_matmul",
    "cmp_epilogue",
    "wr_wait_out",
    "wr_write",
    "wr_final_barrier",
    "rd_reserve",
    "rd_mcast_a",
    "rd_mcast_b",
]:
    if name not in z:
        continue
    cs = z[name]
    sums = [sum(e - s for s, e in v) for v in cs.values()]
    first = [min(s for s, _ in v) - t0 for v in cs.values()]
    last = [max(e for _, e in v) - t0 for v in cs.values()]
    n = st.median(len(v) for v in cs.values())
    print(
        f"{name:18s} n/core={n:5.0f} sum p50={us(st.median(sums)):7.1f} max={us(max(sums)):7.1f} | first p50={us(st.median(first)):7.1f} | lastend p50={us(st.median(last)):7.1f} max={us(max(last)):7.1f}"
    )
# per-block timeline (TRISC0 cmp zones, median over cores)
for name in ["cmp_matmul", "cmp_epilogue"]:
    cs = {c: v for c, v in z[name].items() if c[2].endswith("0") or c[2] == "TRISC_0"}
    if not cs:
        continue
    nb = min(len(v) for v in cs.values())
    for i in range(nb):
        s = st.median(sorted(v)[i][0] - t0 for v in cs.values())
        e = st.median(sorted(v)[i][1] - t0 for v in cs.values())
        print(f"  {name} block{i}: start p50 {us(s):7.1f} end p50 {us(e):7.1f}")
ww = z["wr_write"]
ends = sorted(e - t0 for v in ww.values() for _, e in v)
starts = sorted(s - t0 for v in ww.values() for s, _ in v)
print(
    f"wr_write starts: p10 {us(starts[len(starts)//10]):.1f} p50 {us(starts[len(starts)//2]):.1f} ; ends p90 {us(ends[9*len(ends)//10]):.1f} max {us(ends[-1]):.1f}"
)
