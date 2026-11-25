"""Per-instance durations of one zone (last call per chip): python3 zone_instances.py csv ZONE [RISC]"""
import csv, sys
from collections import defaultdict

path, zone = sys.argv[1], sys.argv[2]
risc = sys.argv[3] if len(sys.argv) > 3 else "TRISC_1"
f = open(path)
f.readline()
rows = list(csv.DictReader(f, skipinitialspace=True))
last = {}
for r in rows:
    last[r["PCIe slot"]] = max(last.get(r["PCIe slot"], 0), int(r["run host ID"]))
for chip in sorted(last):
    t0 = min(
        int(r["time[cycles since reset]"])
        for r in rows
        if r["PCIe slot"] == chip and int(r["run host ID"]) == last[chip]
    )
    ev = defaultdict(list)
    for r in rows:
        if (
            r["PCIe slot"] != chip
            or int(r["run host ID"]) != last[chip]
            or r["zone name"] != zone
            or r["RISC processor type"] != risc
        ):
            continue
        ev[(r["core_x"], r["core_y"])].append((int(r["time[cycles since reset]"]), r["type"]))
    durs, starts = [], []
    for c, e in ev.items():
        e.sort()
        st = None
        for t, ty in e:
            if ty == "ZONE_START":
                st = t
            elif st is not None:
                durs.append((t - st) / 1350.0)
                starts.append((st - t0) / 1350.0)
                st = None
    durs.sort()
    if durs:
        print(
            f"chip {chip}: n={len(durs)} dur min {durs[0]:.2f} med {durs[len(durs)//2]:.2f} max {durs[-1]:.2f}; starts {min(starts):.1f}..{max(starts):.1f}"
        )
