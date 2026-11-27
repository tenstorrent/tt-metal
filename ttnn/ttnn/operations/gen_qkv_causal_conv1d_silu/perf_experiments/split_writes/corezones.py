# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-zone summed duration (us) on chosen cores of one run: python3 corezones.py <csv> <run#> x,y [x,y ...]"""
import csv
import sys
from collections import defaultdict

path, run_idx = sys.argv[1], int(sys.argv[2])
cores = [tuple(int(v) for v in a.split(",")) for a in sys.argv[3:]]
f = open(path)
header = f.readline()
mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0].strip()) if "CHIP_FREQ[MHz]:" in header else 1350.0
rows = list(csv.DictReader(f, skipinitialspace=True))
runs = sorted({int(r["run host ID"]) for r in rows})
run = runs[run_idx]
ev = defaultdict(list)
for r in rows:
    if int(r["run host ID"]) != run:
        continue
    c = (int(r["core_x"]), int(r["core_y"]))
    if c in cores:
        ev[(c, r["RISC processor type"], r["zone name"])].append((int(r["time[cycles since reset]"]), r["type"]))
tot = defaultdict(float)
for k, e in ev.items():
    e.sort()
    st = []
    for t, ty in e:
        if ty == "ZONE_START":
            st.append(t)
        elif st:
            tot[k] += (t - st.pop()) / mhz
names = sorted({(k[1], k[2]) for k in tot})
print("risc     zone                     " + " ".join(f"{str(c):>10s}" for c in cores))
for risc, z in names:
    if z.endswith("FW"):
        continue
    print(f"{risc:8s} {z:24s} " + " ".join(f"{tot.get((c, risc, z), 0):10.1f}" for c in cores))
