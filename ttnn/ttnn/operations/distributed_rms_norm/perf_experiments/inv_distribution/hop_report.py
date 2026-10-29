# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inv distribution hop per (row-group, wave), off a DRMS_PERF_ZONES=1 profile_log_device.csv (last run per chip).

Workers = cores with an rd_inv_col0 zone, sorted (y, x) = the op's row-major worker order; group g = workers
[g*G_w, (g+1)*G_w). Owner of wave w in a group = member (w * owners + k) % G_w (owners = 1 assumed: the D3 focus).
Per (g, w): fold end (owner cmp_fold end) -> allgather start / end (owner) -> last member inv arrival (max over
members of rd_inv_col0 start) -> last member move end -> last member cmp_scale end. All relative to the fold end.

    python3 hop_report.py <csv> <G_w> [chip]
"""

import csv
import statistics
import sys
from collections import defaultdict

log, G_w = sys.argv[1], int(sys.argv[2])
chips_sel = int(sys.argv[3]) if len(sys.argv) > 3 else None
with open(log) as fh:
    mhz = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
by = defaultdict(list)
for r in rows:
    by[int(r["PCIe slot"])].append(r)
agg = defaultdict(list)
for chip, rr in sorted(by.items()):
    if chips_sel is not None and chip != chips_sel:
        continue
    rid = max(int(r["run host ID"]) for r in rr)
    rr = [r for r in rr if int(r["run host ID"]) == rid]
    st, occ = defaultdict(list), defaultdict(list)
    for r in rr:
        k = (int(r["core_x"]), int(r["core_y"]), r["RISC processor type"], r["zone name"])
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            st[k].append(t)
        elif r["type"] == "ZONE_END" and st[k]:
            occ[k].append((st[k].pop(), t))
    for k in occ:
        occ[k].sort()
    ns = lambda c: c * 1000.0 / mhz
    workers = sorted({(k[0], k[1]) for k in occ if k[3] == "rd_inv_col0"}, key=lambda c: (c[1], c[0]))
    groups = [workers[i : i + G_w] for i in range(0, len(workers), G_w)]

    def z(core, zone, i):
        for (x, y, risc, name), v in occ.items():
            if (x, y) == core and name == zone and i < len(v):
                return v[i]
        return None

    kernel_end = max(e for k, v in occ.items() if k[3].endswith("KERNEL") for _, e in v)
    kernel_start = min(s for k, v in occ.items() if k[3].endswith("KERNEL") for s, _ in v)
    print(
        f"chip {chip}: workers {len(workers)} groups {len(groups)} kernel {ns(kernel_end - kernel_start)/1000:.2f} us"
    )
    nw = max(len(occ.get((c[0], c[1], r, "rd_inv_col0"), [])) for c in workers for r in ("NCRISC", "BRISC"))
    for w in range(nw):
        for g, grp in enumerate(groups):
            owner = grp[w % G_w]
            # owner's ordinal of wave w among its owned waves (owners = 1: member j owns waves j, j + G_w, ...)
            o = w // G_w
            fold = z(owner, "cmp_fold", o)
            ag = z(owner, "rd_allgather", o)
            if fold is None or ag is None:
                continue
            mv = [z(c, "rd_inv_col0", w) for c in grp]
            sc = [z(c, "cmp_scale", w) for c in grp]
            if any(m is None for m in mv):
                continue
            t0 = fold[1]
            rel = lambda c: ns(c - t0)
            arr = max(m[0] for m in mv)
            bar = z(owner, "rd_ag_barrier", o)
            row = dict(
                ag_start=rel(ag[0]),
                bar_start=rel(bar[0]) if bar else float("nan"),
                ag_end=rel(ag[1]),
                arrive=rel(arr),
                move_end=rel(max(m[1] for m in mv)),
                move=ns(max(m[1] - m[0] for m in mv)),
                scale_end=rel(max(s[1] for s in sc if s)) if all(sc) else float("nan"),
            )
            for k2, v2 in row.items():
                agg[k2].append(v2)
            if chips_sel is not None:
                print(f"  w{w} g{g} " + " ".join(f"{k2} {v2:7.0f}" for k2, v2 in row.items()) + " ns")
print(
    "MEDIAN over (chip, group, wave) [ns rel. owner fold end]: "
    + " ".join(f"{k}={statistics.median(v):.0f}" for k, v in agg.items())
)
print("MAX: " + " ".join(f"{k}={max(v):.0f}" for k, v in agg.items()))
