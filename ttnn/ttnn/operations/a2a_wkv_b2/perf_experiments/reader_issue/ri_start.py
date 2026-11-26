"""Per chip (last replay): worker reader kernel start and first reader_w_issue start, rel. to instance start."""
import csv, sys
from collections import defaultdict

FREQ = 1350.0
rows = list(csv.reader(open(sys.argv[1])))[2:]
kst = defaultdict(list)
for r in rows:
    if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START":
        kst[r[0]].append(int(r[5]))
for chip in sorted(kst, key=int):
    ts = sorted(kst[chip])
    starts = [ts[0]]
    for a, b in zip(ts, ts[1:]):
        if b - a > 5 * FREQ:
            starts.append(b)
    t0 = starts[-1]
    ks, ws, we, last = [], [], [], []
    for r in rows:
        if r[0] != chip or int(r[5]) < t0:
            continue
        z, typ, risc = r[10].strip(), r[11].strip(), r[3].strip()
        if risc == "NCRISC" and z == "NCRISC-KERNEL" and typ == "ZONE_START":
            ks.append((int(r[5]) - t0) / FREQ)
        if risc == "NCRISC" and z == "reader_w_issue" and typ == "ZONE_START":
            ws.append((int(r[5]) - t0) / FREQ)
        if risc == "NCRISC" and z == "reader_w_issue" and typ == "ZONE_END":
            we.append((int(r[5]) - t0) / FREQ)
    print(
        f"chip {chip}: NCRISC kernel start min {min(ks):.1f} max {max(ks):.1f} | w_issue start {min(ws):.1f}-{max(ws):.1f} end {min(we):.1f}-{max(we):.1f} (n={len(ws)})"
    )
