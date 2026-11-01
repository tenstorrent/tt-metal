"""Per-core phase ends for one run: rows (row_mean us / ~1.03), read end, compute end, write end, write tail.

usage: cores.py [run_index (0-based among runs)] [csv]
"""
import csv, sys
from collections import defaultdict

run_sel = int(sys.argv[1]) if len(sys.argv) > 1 else 0
path = (
    sys.argv[2] if len(sys.argv) > 2 else "/localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv"
)
with open(path) as f:
    header = f.readline()
    freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
runs = []
data = defaultdict(lambda: defaultdict(dict))
open_ = {}
for r in rows[1:]:
    r = [c.strip() for c in r]
    run = r[idx["run host ID"]]
    if run not in runs:
        runs.append(run)
    core = (int(r[idx["core_x"]]), int(r[idx["core_y"]]))
    risc = r[idx["RISC processor type"]]
    z = r[idx["zone name"]]
    t = int(r[idx["time[cycles since reset]"]])
    k = (run, core, risc, z)
    if r[idx["type"]] == "ZONE_START":
        open_[k] = t
        d = data[run][core]
        d.setdefault("t0", t)
        d["t0"] = min(d["t0"], t)
        if risc == "TRISC_1" and z == "compute_row_mean":
            d["blocks"] = d.get("blocks", 0) + 1
    elif k in open_:
        s = open_.pop(k)
        d = data[run][core]
        if risc == "TRISC_1" and z == "compute_row_mean":
            d["mean_ns"] = d.get("mean_ns", 0) + (t - s) / freq * 1000
        if z.endswith("-KERNEL"):
            d[risc] = t
run = runs[run_sel]
cores = data[run]
t0 = min(d["t0"] for d in cores.values())
us = lambda c: (c - t0) / freq
out = []
for core, d in cores.items():
    out.append((us(d["BRISC"]), core, round(d.get("mean_ns", 0) / 1030), us(d["NCRISC"]), us(d["TRISC_1"])))
out.sort(reverse=True)
print(f"run {run}: core rows read_end compute_end write_end tail")
for item in out[:15] + out[len(out) // 2 : len(out) // 2 + 5] + out[-5:]:
    w, core, nrows, rd, cp = item
    print(f"{core} rows~{nrows} read {rd:5.1f} compute {cp:5.1f} write {w:5.1f} tail {w-cp:5.1f}")
by = defaultdict(list)
for w, core, nrows, rd, cp in out:
    by[nrows].append((rd, cp, w))
for n, v in sorted(by.items()):
    m = lambda i: sum(x[i] for x in v) / len(v)
    print(
        f"rows~{n}: {len(v)} cores avg read {m(0):.1f} compute {m(1):.1f} write {m(2):.1f}; max write {max(x[2] for x in v):.1f}"
    )
