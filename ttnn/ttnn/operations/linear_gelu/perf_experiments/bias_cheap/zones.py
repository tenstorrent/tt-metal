"""Per-run (one op per test, in order) math-thread (TRISC_1) compute_matmul_gelu zones on each run's busiest core.
Usage: zones.py <profile_log_device.csv> <order.log>"""
import csv
import sys
from collections import defaultdict

path, order = sys.argv[1], sys.argv[2]
f = open(path)
freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rows = list(csv.DictReader(f, skipinitialspace=True))
runs = sorted({int(r["run host ID"]) for r in rows})
lines = [l.strip() for l in open(order)][-len(runs) :]
for rid, line in zip(runs, lines):
    rr = [r for r in rows if int(r["run host ID"]) == rid]
    t0 = min(int(r["time[cycles since reset]"]) for r in rr)
    open_, ev = {}, defaultdict(list)
    for r in rr:
        k = ((r["core_x"], r["core_y"]), r["RISC processor type"], r["zone name"])
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            open_[k] = t
        elif k in open_:
            ev[k[0]].append((k[1], k[2], (open_.pop(k) - t0) / freq, (t - t0) / freq))
    end = {c: max(e[3] for e in v) for c, v in ev.items()}
    core = max(end, key=end.get)
    mm = [e for e in ev[core] if e[0] == "TRISC_1" and e[1] == "compute_matmul_gelu"]
    mm.sort(key=lambda e: e[2])
    durs = [round(e[3] - e[2], 2) for e in mm]
    print(f"{line.split('|ulp')[0]} core={core} end={end[core]:.1f} math_rb_us={durs} sum={sum(durs):.2f}")
