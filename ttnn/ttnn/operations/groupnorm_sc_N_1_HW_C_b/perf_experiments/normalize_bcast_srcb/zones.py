#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per run: p50 / max over cores of one zone's per-core summed duration (ns), per TRISC.
usage: zones.py <profile_log_device.csv> <runs.jsonl or results_*.json> [zone=c_pass2]"""
import csv, json, sys
from collections import defaultdict
from statistics import median

path, runsp = sys.argv[1], sys.argv[2]
zone = sys.argv[3] if len(sys.argv) > 3 else "c_pass2"
runs = json.load(open(runsp)) if runsp.endswith(".json") else [json.loads(l) for l in open(runsp) if l.strip()]
with open(path) as f:
    head = f.readline()
    mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(f, skipinitialspace=True))
k = 1000.0 / mhz
st = {}
dur = defaultdict(lambda: defaultdict(float))  # (run, risc) -> core -> ns
ids = []
for r in rows:
    rid = r["run host ID"]
    if rid not in ids:
        ids.append(rid)
    if r["zone name"] != zone:
        continue
    key = (rid, r["RISC processor type"], r["core_x"], r["core_y"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        st[key] = t
    elif r["type"] == "ZONE_END" and key in st:
        dur[(rid, key[1])][(key[2], key[3])] += (t - st.pop(key)) * k
ids = [i for i in ids if any((i, rr) in dur for rr in ("TRISC_0", "TRISC_1", "TRISC_2"))]
print(f"{'variant':26s} {'case':28s} " + "  ".join(f"{r+' p50/max':>18s}" for r in ("TRISC_0", "TRISC_1", "TRISC_2")))
for run, rid in zip(runs, ids):
    cells = []
    for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
        v = list(dur[(rid, risc)].values())
        cells.append(f"{median(v):8.0f}/{max(v):8.0f}" if v else f"{'-':>17s}")
    print(f"{run['variant'][-26:]:26s} {run['case']:28s} " + "  ".join(f"{c:>18s}" for c in cells))
