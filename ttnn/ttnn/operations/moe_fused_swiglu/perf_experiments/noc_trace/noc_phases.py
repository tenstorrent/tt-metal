#!/usr/bin/env python3
"""Phase-labelled NoC timeline + DRAM endpoint balance, from a tt-metal noc_trace json.

NOTE: payload_chunks is a saturating uint8 of 32B chunks -> any transfer >= 8160 B
is reported as exactly 8160.  Reads here are 576-2304 B so they are exact; multicasts
are all saturated and are re-priced analytically by the caller.
"""
import json, sys
from collections import Counter, defaultdict

path = sys.argv[1]
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FREQ = 1.35

ev = json.load(open(path))
t0 = min(e["timestamp"] for e in ev)
t1 = max(e["timestamp"] for e in ev)
us = lambda t: (t - t0) / FREQ / 1000.0

xfer = [e for e in ev if e.get("type") in ("READ", "WRITE_", "WRITE_MULTICAST")]
sat = sum(1 for e in xfer if e["num_bytes"] == 8160)
print(f"span {us(t1):.1f} us, {len(xfer)} transfers, {sat} saturated (>=8160B, true size unknown)")

cores = {(e["sx"], e["sy"]) for e in ev}
dram = {(e["dx"], e["dy"]) for e in xfer if "dx" in e} - cores

# ---- zone occupancy over time (which stage are the cores in?) ----
open_z = defaultdict(list)
spans = defaultdict(list)
for e in sorted(ev, key=lambda e: e["timestamp"]):
    if "zone" not in e:
        continue
    k = (e["sx"], e["sy"], e["proc"], e["zone"])
    if e["zone_phase"] == "ZONE_START":
        open_z[k].append(e["timestamp"])
    elif open_z[k]:
        spans[e["zone"]].append((open_z[k].pop(), e["timestamp"]))

ZONES = [z for z in spans if not z.endswith("-KERNEL") and not z.startswith("PROFILER")]
NCORE = len(cores)
w = (t1 - t0) / NB

print(f"\n--- {NB} buckets of {w/FREQ:.0f} ns.  cells = % of {NCORE} cores inside the stage ---")
hdr = "".join(f"{int(us(t0+i*w)):>4d}" for i in range(NB))
occ = {}
for z in ZONES:
    row = [0.0] * NB
    for b, e in spans[z]:
        for i in range(NB):
            lo, hi = t0 + i * w, t0 + (i + 1) * w
            ov = min(e, hi) - max(b, lo)
            if ov > 0:
                row[i] += ov / w
    occ[z] = row
print(f"{'stage':20s} {hdr}")
for z in sorted(ZONES, key=lambda z: -sum(occ[z])):
    if sum(occ[z]) < 0.3:
        continue
    line = "".join("    " if occ[z][i] / NCORE < 0.02 else f"{int(occ[z][i]/NCORE*100):>4d}" for i in range(NB))
    print(f"{z:20s} {line}")

# ---- DRAM read rate + endpoint balance over time ----
print(f"\n{'stage':20s} {hdr}")
for label, pred in (
    ("DRAM rd GB/s", lambda e: e["type"] == "READ" and (e.get("dx"), e.get("dy")) in dram),
    ("L1 rd GB/s", lambda e: e["type"] == "READ" and (e.get("dx"), e.get("dy")) not in dram),
    ("uni wr GB/s", lambda e: e["type"] == "WRITE_"),
):
    row = [0.0] * NB
    for e in xfer:
        if pred(e):
            i = min(NB - 1, int((e["timestamp"] - t0) / w))
            row[i] += e["num_bytes"]
    line = "".join(f"{int(v/(w/FREQ)):>4d}" for v in row)
    print(f"{label:20s} {line}")

# multicast issue count per bucket
row = [0] * NB
for e in xfer:
    if e["type"] == "WRITE_MULTICAST":
        row[min(NB - 1, int((e["timestamp"] - t0) / w))] += 1
print(f"{'mcasts issued':20s} " + "".join(f"{v:>4d}" if v else "    " for v in row))


# ---- DRAM endpoint balance, whole op and per phase ----
def bank_table(lo, hi, title):
    b = Counter()
    n = Counter()
    for e in xfer:
        if e["type"] == "READ" and (e.get("dx"), e.get("dy")) in dram and lo <= e["timestamp"] < hi:
            b[(e["dx"], e["dy"])] += e["num_bytes"]
            n[(e["dx"], e["dy"])] += 1
    if not b:
        return
    tot = sum(b.values())
    mx = max(b.values())
    mean = tot / len(dram)
    dur = (hi - lo) / FREQ
    print(
        f"\n{title}: {tot/1e6:.3f} MB over {dur:,.0f} ns = {tot/dur:.0f} GB/s "
        f"| {len(b)}/{len(dram)} endpoints live, max/mean = {mx/mean:.2f}"
    )
    for d in sorted(dram):
        bar = "#" * int(b[d] / max(1, mx) * 40)
        print(f"   {str(d):9s} {b[d]/1e6:7.3f} MB {n[d]:6d} txn {b[d]/max(1,n[d]):6.0f} B  {bar}")


bank_table(t0, t1, "ALL DRAM reads")
sys.stdout.flush()
