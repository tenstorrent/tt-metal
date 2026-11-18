# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-owner slice arrival times of an N-slice-outer K split dispatch (no device).

    python3 arrivals.py <profile_log_device.csv> <run id> <k_cols> [--own-local]

Maps each core's writer/reader send zones (in walk order) to their owner (conv2d_3x3_nslice_order.hpp) and prints,
for the owners whose combine wait was longest, every peer's send end (us from the first kernel start)."""
import argparse, collections, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import zone_report
from timeline import spans


def owner(k, my, j):
    if j == k - 1:
        return my
    d = j // 2 + 1
    return (my + d) % k if j % 2 == 0 else (my + k - d) % k


ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("run", type=int)
ap.add_argument("k", type=int)
ap.add_argument("--own-local", action="store_true")
ap.add_argument("--top", type=int, default=4)
a = ap.parse_args()
freq, runs = zone_report.load(a.log)
ev = runs[a.run]
sp = spans(ev)
k0 = min(s for (c, r, z), v in sp.items() if z.endswith("-KERNEL") for s, _ in v)
us = lambda t: (t - k0) / freq
xs = sorted({c[0] for c, *_ in ev})
col = {x: i for i, x in enumerate(xs)}
arr = collections.defaultdict(dict)  # owner core -> {sender col: (risc, end)}
for c in {c for c, *_ in ev}:
    my = col[c[0]]
    w = [s for s in sp.get((c, "BRISC", "writer_combine_send"), [])]
    rd = [s for s in sp.get((c, "NCRISC", "reader_combine_send"), [])]
    wj = [j for j in range(a.k) if not (j != a.k - 1 and j % 2 == 0) and not (a.own_local and j == a.k - 1)]
    rj = [j for j in range(a.k) if j != a.k - 1 and j % 2 == 0]
    for lst, js, risc in ((w, wj, "W"), (rd, rj, "R")):
        for (s, e), j in zip(sorted(lst), js):
            o = owner(a.k, my, j)
            arr[(xs[o], c[1])][my] = (risc, j, us(s), us(e))
waits = []
for c in {c for c, *_ in ev}:
    v = sp.get((c, "TRISC_0", "compute_combine_wait"))
    if v:
        waits.append((us(v[-1][1]) - us(v[-1][0]), c, us(v[-1][0]), us(v[-1][1])))
for wlen, c, ws, we in sorted(waits, reverse=True)[: a.top]:
    print(f"owner {c} col {col[c[0]]}: wait {ws:.2f}..{we:.2f} ({wlen:.2f})")
    for sc, (risc, j, s, e) in sorted(arr[c].items(), key=lambda kv: kv[1][3]):
        print(f"    from col {sc} ({risc} step {j}): send {s:.2f}..{e:.2f}")
