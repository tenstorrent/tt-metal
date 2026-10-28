# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""usage: python3 spread.py [report_dir] -> per generic op call: per-core BRISC/NCRISC kernel end (us after
the earliest kernel start) p10/p50/max, plus the slowest cores."""
import csv
import glob
import sys
from collections import defaultdict

MHZ = 1350.0
rep = (
    sys.argv[1]
    if len(sys.argv) > 1
    else sorted(glob.glob("/localdev/jmitrovic/tt-metal/generated/profiler/reports/*/"))[-1]
)
d = defaultdict(dict)
with open(f"{rep}/profile_log_device.csv") as f:
    f.readline()
    r = csv.reader(f)
    next(r)
    for row in r:
        if len(row) < 12:
            continue
        x, y, risc, t, call, zone, typ = (
            int(row[1]),
            int(row[2]),
            row[3].strip(),
            int(row[5]),
            row[7].strip(),
            row[10].strip(),
            row[11].strip(),
        )
        if zone.endswith("-KERNEL"):
            k = (x, y, risc, typ)
            dd = d[call]
            dd[k] = min(dd.get(k, t), t) if typ == "ZONE_START" else max(dd.get(k, t), t)
for call in sorted(d, key=lambda c: int(c)):
    dd = d[call]
    t0 = min(v for k, v in dd.items() if k[3] == "ZONE_START")
    for risc in ("BRISC", "NCRISC"):
        ends = sorted(((v - t0) / MHZ, k[0], k[1]) for k, v in dd.items() if k[2] == risc and k[3] == "ZONE_END")
        if not ends:
            continue
        e = [a for a, _, _ in ends]
        print(
            f"call {call} {risc:6s} n={len(e)} end p10={e[len(e)//10]:.1f} p50={e[len(e)//2]:.1f} max={e[-1]:.1f} slowest={[(round(a,1),x,y) for a,x,y in ends[-3:]]}"
        )
