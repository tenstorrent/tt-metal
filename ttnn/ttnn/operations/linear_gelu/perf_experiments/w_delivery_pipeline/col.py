"""Usage: col.py <csv> <run index> <phys x>: per core in the column, sender-side W zones (rdbar, send) per round."""
import csv, sys
from collections import defaultdict

f = open(sys.argv[1])
first = f.readline()
freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rows = list(csv.DictReader(f, skipinitialspace=True))
ids = sorted({int(r["run host ID"]) for r in rows})
rid = ids[int(sys.argv[2])]
rows = [r for r in rows if int(r["run host ID"]) == rid and r["RISC processor type"] == "NCRISC"]
t0 = min(int(r["time[cycles since reset]"]) for r in rows)
op = {}
ev = []
for r in rows:
    if r["core_x"] != sys.argv[3]:
        continue
    k = (r["core_y"], r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        op[k] = t
    elif k in op and k[1] in ("w_rdbar", "w_send"):
        ev.append((int(k[0]), k[1], (op.pop(k) - t0) / freq, (t - t0) / freq))
for e in sorted(ev, key=lambda e: e[2]):
    print(f"y={e[0]:2d} {e[1]:8s} {e[2]:7.2f} -> {e[3]:7.2f} ({e[3]-e[2]:5.2f})")
