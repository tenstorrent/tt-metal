"""Per-time-bin count of cores inside each zone (all cores), for one run host id.
usage: python3 occupancy.py zones.csv run_id [bin_us] [zones,comma]"""
import csv, sys
from collections import defaultdict

path, run = sys.argv[1], sys.argv[2]
BIN = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
Z = (
    sys.argv[4] if len(sys.argv) > 4 else "reader_barrier,writer_issue,writer_barrier,compute_mix,writer_help_read"
).split(",")
MHZ = 1350.0
ev = defaultdict(list)
t0 = None
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for row in r:
        row = [x.strip() for x in row]
        if row[7] != run:
            continue
        t = int(row[5])
        t0 = t if t0 is None else min(t0, t)
        ev[(row[1], row[2], row[3], row[10])].append((t, row[11]))
occ = defaultdict(lambda: defaultdict(float))
tmax = 0
for (x, y, risc, z), lst in ev.items():
    if z not in Z:
        continue
    if z == "compute_mix" and risc != "TRISC_0":
        continue
    lst.sort()
    s = None
    for t, ty in lst:
        if ty == "ZONE_START":
            s = t
        elif s is not None:
            a, b = (s - t0) / MHZ, (t - t0) / MHZ
            tmax = max(tmax, b)
            k = int(a // BIN)
            while k * BIN < b:
                lo, hi = max(a, k * BIN), min(b, (k + 1) * BIN)
                occ[z][k] += (hi - lo) / BIN
                k += 1
            s = None
print("t_us  " + " ".join(f"{z[:14]:>14s}" for z in Z))
for k in range(int(tmax // BIN) + 1):
    print(f"{k*BIN:5.0f} " + " ".join(f"{occ[z][k]:14.1f}" for z in Z))
