#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate DRAM throughput over time for one DRAM-interleaved run, reconstructed from per-core zone intervals:
each core's pass-1 bytes (reads) spread uniformly over its TRISC_1 c_pass1 zone, pass-2 bytes (re-reads + writes)
over its c_pass2 zone. Prints active cores and GB/s per time bin. A model, not a measurement of the DRAM itself.
usage: agg_bw.py <profile_log_device.csv> <run host ID> <B> <K> <h> <W> [bins]"""
import csv, sys
from collections import defaultdict

path, run = sys.argv[1], sys.argv[2]
B, K, h, W = map(int, sys.argv[3:7])
bins = int(sys.argv[7]) if len(sys.argv) > 7 else 20
with open(path) as f:
    head = f.readline()
    mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = [
        r
        for r in csv.DictReader(f, skipinitialspace=True)
        if r["run host ID"] == run and r["RISC processor type"] == "TRISC_1"
    ]
iv = defaultdict(dict)
for r in rows:
    if r["zone name"] in ("c_pass1", "c_pass2"):
        c = (r["core_x"], r["core_y"])
        iv[(c, r["zone name"])][r["type"]] = int(r["time[cycles since reset]"])
t0 = min(v["ZONE_START"] for v in iv.values())
t1 = max(v["ZONE_END"] for v in iv.values())
chunk = h * W * 2048
p1_bytes = B * chunk
p2_bytes = (2 * B - K) * chunk  # writes B chunks + re-reads (B-K)
edges = [t0 + (t1 - t0) * i / bins for i in range(bins + 1)]
agg = [0.0] * bins
act = [0] * bins
for (c, z), v in iv.items():
    s, e = v["ZONE_START"], v["ZONE_END"]
    by = p1_bytes if z == "c_pass1" else p2_bytes
    rate = by / max(e - s, 1)
    for i in range(bins):
        ov = max(0, min(e, edges[i + 1]) - max(s, edges[i]))
        agg[i] += rate * ov
        if ov > 0:
            act[i] += 1
dt_ns = (t1 - t0) / bins * 1000 / mhz
tot = sum(agg)
for i in range(bins):
    print(f"{(edges[i]-t0)*1e-3/mhz*1000:8.0f}us  active {act[i]:4d}  {agg[i]/dt_ns:6.0f} GB/s")
print(f"total {tot/1e6:.0f} MB over {(t1-t0)/mhz:.0f} us -> {tot/((t1-t0)*1000/mhz):.0f} GB/s mean")
