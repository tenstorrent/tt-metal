"""Per-row mechanism check from profile_log_device.csv: output-stage duration, write tail after it, op end.

Usage: mech.py <whole_result_dir> [row ...]. Output stage = first compute_normalize / compute_output_chunked start
to last compute_affine_shift / compute_output_chunked end on the core's LAST block; write tail = writer kernel's
last zone end minus that output-stage end. us at CHIP_FREQ.
"""

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

res = Path(sys.argv[1])
want = sys.argv[2:]
ids = (res / "ids.txt").read_text().split()
with open(res / "profile_log_device.csv") as f:
    header = f.readline()
    freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
idx = {c.strip(): i for i, c in enumerate(rows[0])}
zones = defaultdict(list)  # (run, core, zone) -> [(start, end)]
op_start = {}
open_ = {}
for r in rows[1:]:
    r = [c.strip() for c in r]
    run, core, zone, typ = (
        r[idx["run host ID"]],
        (r[idx["core_x"]], r[idx["core_y"]]),
        r[idx["zone name"]],
        r[idx["type"]],
    )
    t = int(r[idx["time[cycles since reset]"]])
    op_start[run] = min(op_start.get(run, t), t)
    k = (run, core, zone)
    if typ == "ZONE_START":
        open_[k] = t
    elif typ == "ZONE_END" and k in open_:
        zones[k].append((open_.pop(k), t))
runs = sorted({k[0] for k in zones}, key=int)
us = lambda c: c / freq
for run, name in zip(runs, ids):
    if want and name not in want:
        continue
    cores = {k[1] for k in zones if k[0] == run}
    dur, tail, wend = [], [], []
    for core in cores:
        z = lambda n: zones.get((run, core, n), [])
        if z("compute_output_chunked"):
            s, e = z("compute_output_chunked")[-1]
        elif z("compute_normalize"):
            s, e = z("compute_normalize")[-1][0], z("compute_affine_shift")[-1][1]
        else:
            continue
        w = z("writer_final_barrier")
        if not w:
            continue
        dur.append(us(e - s))
        tail.append(us(w[-1][1] - e))
        wend.append(us(w[-1][1] - op_start[run]))
    print(
        f"{name:30s} cores={len(dur):3d} out_stage med={st.median(dur):.2f}us  write_tail med={st.median(tail):.2f} "
        f"max={max(tail):.2f}us  writer_end med={st.median(wend):.2f} max={max(wend):.2f}us"
    )
