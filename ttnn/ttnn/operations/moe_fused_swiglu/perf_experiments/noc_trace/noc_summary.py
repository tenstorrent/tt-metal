#!/usr/bin/env python3
"""Analyse a tt-metal noc_trace_*.json: who talks to whom, when, and how much."""
import json, sys
from collections import Counter, defaultdict

path = sys.argv[1]
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FREQ = 1.35  # GHz -> ns = cycles / FREQ

ev = json.load(open(path))
xfer = [e for e in ev if e.get("type") in ("READ", "WRITE_", "WRITE_MULTICAST")]
zones = [e for e in ev if "zone" in e]

t0 = min(e["timestamp"] for e in ev)
t1 = max(e["timestamp"] for e in ev)
T = (t1 - t0) / FREQ
print(f"{len(ev)} events, {len(xfer)} transfers, span {T:,.0f} ns")

cores = sorted({(e["sx"], e["sy"]) for e in ev})
print(
    f"{len(cores)} source cores  x:{min(c[0] for c in cores)}-{max(c[0] for c in cores)}"
    f"  y:{min(c[1] for c in cores)}-{max(c[1] for c in cores)}"
)

dsts = Counter()
dbytes = Counter()
for e in xfer:
    if "dx" in e:
        dsts[(e["dx"], e["dy"])] += 1
        dbytes[(e["dx"], e["dy"])] += e["num_bytes"]
print("\n--- top destinations by BYTES (dx,dy): txns, MB, mean B/txn ---")
for d, b in dbytes.most_common(20):
    print(f"  {str(d):10s} {dsts[d]:7d} txn  {b/1e6:8.3f} MB  {b/max(1,dsts[d]):7.0f} B")

# DRAM cores = destinations only ever read from, outside the compute grid coords seen as sources
core_set = set(cores)
dram = {d for d in dbytes if d not in core_set}
print(f"\ncompute-grid source coords: {sorted(core_set)[:4]} ... ({len(core_set)})")
print(f"non-grid (DRAM/other) dst coords: {sorted(dram)}")

tot_dram = sum(b for d, b in dbytes.items() if d in dram)
tot_l1 = sum(b for d, b in dbytes.items() if d not in dram)
print(f"\nbytes to DRAM coords: {tot_dram/1e6:.3f} MB   to grid L1: {tot_l1/1e6:.3f} MB")

print("\n--- per NoC ---")
noc_b = Counter()
noc_n = Counter()
for e in xfer:
    noc_b[e["noc"]] += e["num_bytes"]
    noc_n[e["noc"]] += 1
for n in sorted(noc_b):
    print(f"  {n}: {noc_n[n]:7d} txn  {noc_b[n]/1e6:8.3f} MB")

print("\n--- per proc ---")
p_b = Counter()
p_n = Counter()
for e in xfer:
    p_b[e["proc"]] += e["num_bytes"]
    p_n[e["proc"]] += 1
for n in sorted(p_b):
    print(f"  {n}: {p_n[n]:7d} txn  {p_b[n]/1e6:8.3f} MB")

# ---- timeline ----
w = (t1 - t0) / NB
print(f"\n--- timeline, {NB} buckets of {w/FREQ:,.0f} ns ---")
print(f"{'t(us)':>7} {'DRAMrd GB/s':>12} {'L1rd GB/s':>11} {'wr GB/s':>9} {'txn/us':>8}  bar")
for i in range(NB):
    lo, hi = t0 + i * w, t0 + (i + 1) * w
    dr = l1r = wr = 0
    n = 0
    for e in xfer:
        if lo <= e["timestamp"] < hi:
            n += 1
            d = (e.get("dx"), e.get("dy"))
            if e["type"] == "READ":
                if d in dram:
                    dr += e["num_bytes"]
                else:
                    l1r += e["num_bytes"]
            else:
                wr += e["num_bytes"]
    ns = w / FREQ
    f = lambda b: b / ns  # bytes/ns == GB/s
    bar = "#" * int(f(dr + l1r + wr) / 20)
    print(f"{(lo-t0)/FREQ/1000:7.1f} {f(dr):12.1f} {f(l1r):11.1f} {f(wr):9.1f} {n/ns*1000:8.1f}  {bar}")
