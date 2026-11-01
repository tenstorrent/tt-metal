"""focus2 critical-path breakdown per run: 2-row cores' row-1 read window vs the 1-row cores' output production.

usage: focus2_paths.py <dev.csv>... (run index 1 = focus2 in test_peel order)
"""
import csv
import statistics as st
import sys
from collections import defaultdict


def load(path, run_sel=1):
    with open(path) as f:
        freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.reader(f))
    idx = {c.strip(): i for i, c in enumerate(rows[0])}
    runs, zones, open_ = [], defaultdict(list), {}
    for r in rows[1:]:
        r = [c.strip() for c in r]
        run = r[idx["run host ID"]]
        if run not in runs:
            runs.append(run)
        k = (run, (int(r[idx["core_x"]]), int(r[idx["core_y"]])), r[idx["RISC processor type"]], r[idx["zone name"]])
        t = int(r[idx["time[cycles since reset]"]])
        if r[idx["type"]] == "ZONE_START":
            open_[k] = t
        elif k in open_:
            zones[k].append((open_.pop(k), t))
    run = runs[run_sel]
    per = defaultdict(dict)
    for (rn, core, risc, z), v in zones.items():
        if rn == run:
            per[core][(risc, z)] = v
    t0 = min(s for c in per.values() for v in c.values() for s, _ in v)
    us = lambda t: (t - t0) / freq
    return per, us


OUT_ZONES = ("compute_affine_shift", "compute_affine")
for path in sys.argv[1:]:
    per, us = load(path)
    two = [c for c, z in per.items() if len(z.get(("TRISC_1", "compute_row_mean"), [])) == 2]
    one = [c for c in per if c not in two]
    r1s = [us(per[c][("NCRISC", "reader_block")][-1][0]) for c in two]
    r1e = [us(per[c][("NCRISC", "reader_block")][-1][1]) for c in two]
    kz = lambda c: next(z for z in OUT_ZONES if ("TRISC_1", z) in per[c])
    out_start = [us(per[c][("TRISC_1", kz(c))][0][0]) for c in one]
    end = [us(per[c][("BRISC", "BRISC-KERNEL")][0][1]) for c in per]
    two_tail = [
        us(per[c][("TRISC_1", "TRISC-KERNEL")][0][1]) - us(per[c][("NCRISC", "reader_block")][-1][1]) for c in two
    ]
    # 1-row cores still writing during the 2-row cores' row-1 read window
    w_lo, w_hi = st.mean(r1s), st.mean(r1e)
    print(
        f"{path.split('/')[-1][:24]:24s} 2row: r1 read {st.mean(r1s):5.1f}->{st.mean(r1e):5.1f} ({st.mean(r1e)-st.mean(r1s):4.1f})"
        f" tail {st.mean(two_tail):4.1f} | 1row out_start med {st.median(out_start):5.1f} p10 {sorted(out_start)[len(out_start)//10]:5.1f}"
        f" | op end {max(end):5.1f}"
    )
