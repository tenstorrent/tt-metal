# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core tail: walk end, combine start/end (TRISC_2), store end, kernel end (us after the run's first kernel start),
the N latest-ending cores, plus distribution summaries.
    python3 tail_cores.py <profile_log_device.csv> <run_index> [N]"""
import collections
import csv
import sys

path, idx = sys.argv[1], int(sys.argv[2])
nshow = int(sys.argv[3]) if len(sys.argv) > 3 else 12
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rows:
        if len(r) < len(cols):
            continue
        runs[int(r[ix["run host ID"]])].append(
            (
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
t0 = min(t for c, r, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
us = lambda c: (c - t0) / freq
last = collections.defaultdict(dict)  # core -> key -> (start, end) of the LAST instance
stack = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    key = (core, risc, z)
    if kind == "ZONE_START":
        stack[key].append(t)
    elif stack[key]:
        s = stack[key].pop()
        last[core][(risc, z)] = (us(s), us(t))
recs = []
for core, d in last.items():
    kend = max(v[1] for (r, z), v in d.items() if z.endswith("-KERNEL"))
    g = lambda r, z, i: d.get((r, z), (float("nan"),) * 2)[i]
    recs.append(
        (
            kend,
            core,
            g("TRISC_2", "compute_matmul_walk", 1),
            g("TRISC_2", "compute_combine", 0),
            g("TRISC_2", "compute_combine", 1),
            g("BRISC", "writer_store", 1),
            g("BRISC", "BRISC-KERNEL", 1),
            g("NCRISC", "NCRISC-KERNEL", 1),
            g("TRISC_2", "TRISC-KERNEL", 1),
        )
    )
recs.sort(reverse=True)
print(f"run {big[idx-1]}: {len(recs)} cores")
print("   core     kend   walk_end comb_s comb_e store_e  brisc  ncrisc trisc2  [comb  store-comb  kend-store]")
for r in recs[:nshow]:
    kend, core, we, cs, ce, se, be, ne, te = r
    print(
        f"  {str(core):9} {kend:6.2f}   {we:6.2f} {cs:6.2f} {ce:6.2f} {se:6.2f}  {be:6.2f} {ne:6.2f} {te:6.2f}"
        f"  [{ce-cs:5.2f} {se-ce:5.2f} {kend-se:5.2f}]"
    )


def pct(vals):
    v = sorted(x for x in vals if x == x)
    return f"p10 {v[len(v)//10]:.2f} p50 {v[len(v)//2]:.2f} p90 {v[9*len(v)//10]:.2f} max {v[-1]:.2f}" if v else "-"


print("walk end   ", pct([r[2] for r in recs]))
print("comb dur   ", pct([r[4] - r[3] for r in recs]))
print("store-comb ", pct([r[5] - r[4] for r in recs]))
print("kend-store ", pct([r[0] - r[5] for r in recs]))
print("kend-walk  ", pct([r[0] - r[2] for r in recs]))
print("kernel end ", pct([r[0] for r in recs]))
