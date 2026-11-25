"""Per-row stage-3 timeline of chosen cores (host script; zone CSV of the q_stem_diag package).

    python3 row_table.py zones.csv chip x,y [x,y ...]
Prints per row (consumption order): reader wait end (RW_ROW: the row landed / RW_LOCAL: the own quarter landed),
compute row start / end (CR_ROW, TRISC_1) and writer row end (WR_ROW).
"""
import csv
import sys
from collections import defaultdict

path, chip = sys.argv[1], sys.argv[2]
want = [tuple(int(v) for v in a.split(",")) for a in sys.argv[3:]]
f = open(path)
f.readline()
rows = list(csv.DictReader(f, skipinitialspace=True))
MHZ = 1350.0
lastid = max(int(r["run host ID"]) for r in rows if r["PCIe slot"] == chip)
xs = sorted({int(r["core_x"]) for r in rows if r["PCIe slot"] == chip})
ys = sorted({int(r["core_y"]) for r in rows if r["PCIe slot"] == chip})
ev = defaultdict(list)
t0 = None
for r in rows:
    if r["PCIe slot"] != chip or int(r["run host ID"]) != lastid:
        continue
    c = (xs.index(int(r["core_x"])), ys.index(int(r["core_y"])))
    t = int(r["time[cycles since reset]"])
    if r["zone name"].endswith("KERNEL") and r["type"] == "ZONE_START":
        t0 = t if t0 is None else min(t0, t)
    ev[(c, r["RISC processor type"], r["zone name"], r["type"])].append(t)
for c in want:
    g = lambda risc, z, ty: [(v - t0) / MHZ for v in sorted(ev.get((c, risc, z, ty), []))]
    rw = g("NCRISC", "RW_ROW", "ZONE_END")
    rl = g("NCRISC", "RW_LOCAL", "ZONE_END")
    cs, ce = g("TRISC_1", "CR_ROW", "ZONE_START"), g("TRISC_1", "CR_ROW", "ZONE_END")
    we = g("BRISC", "WR_ROW", "ZONE_END")
    print(f"chip {chip} core {c}: remote-row landings {['%.1f' % v for v in rw]} local {['%.1f' % v for v in rl]}")
    for i in range(len(ce)):
        print(
            f"   row {i:2d} compute {cs[i]:6.1f}-{ce[i]:6.1f} ({ce[i] - cs[i]:4.2f})  write end {we[i] if i < len(we) else float('nan'):6.1f}"
        )
