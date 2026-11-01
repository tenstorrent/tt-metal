"""Per run (one layer_norm call): per-core grid of first reader_block us / NCRISC end us / BRISC end us.

usage: grid_runs.py [profile_log_device.csv] [run_index ...]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "last_profile_log_device.csv")
want = [int(a) for a in sys.argv[2:]]
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
runs = defaultdict(list)
for r in rows[1:]:
    runs[r[idx["run host ID"]].strip()].append([c.strip() for c in r])
run_ids = sorted(runs, key=int)
for n, rid in enumerate(run_ids):
    if want and n not in want:
        continue
    open_, first, ends = {}, {}, defaultdict(dict)
    t0 = None
    for r in runs[rid]:
        core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
        risc, z, typ = r[idx["RISC processor type"]], r[idx["zone name"]], r[idx["type"]]
        t = int(r[idx["time[cycles since reset]"]])
        k = (core, risc, z)
        if typ == "ZONE_START":
            open_[k] = t
            if z.endswith("-KERNEL"):
                t0 = t if t0 is None else min(t0, t)
        elif typ == "ZONE_END" and k in open_:
            s = open_.pop(k)
            if z == "reader_block" and core not in first:
                first[core] = t - s
            if z.endswith("-KERNEL"):
                ends[core][risc] = t
    if not ends:
        continue
    xs = sorted({c[0] for c in ends})
    ys = sorted({c[1] for c in ends})
    us = lambda c: c / freq
    print(f"run #{n} (host id {rid}): first reader_block / NCRISC end / BRISC end (us)")
    for y in ys:
        cells = []
        for x in xs:
            e = ends.get((x, y))
            if not e:
                cells.append(" " * 16)
                continue
            cells.append(
                f"{us(first.get((x, y), 0)):4.1f}/{us(e.get('NCRISC', t0) - t0):4.1f}/{us(e.get('BRISC', t0) - t0):4.1f}"
            )
        print(f"y={y:2d} " + " ".join(cells))
    print("x:", xs)
    allend = max(max(v.values()) for v in ends.values())
    print(f"span {us(allend - t0):.1f} us")
