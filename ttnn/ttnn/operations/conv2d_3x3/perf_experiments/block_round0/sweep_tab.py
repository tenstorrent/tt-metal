# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: per (column, sender row) the column sender's round-0 read (issue end -> landed, us) of a sender-row sweep.
    python3 sweep_tab.py <log> <first run#> <shifts comma list> [k_cols] [rows]"""
import sys

from r0view import XMAP, load, spans

freq, runs = load(sys.argv[1])
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
first = int(sys.argv[2])
shifts = [int(s) for s in sys.argv[3].split(",")]
k_cols = int(sys.argv[4]) if len(sys.argv) > 4 else 8
rows = int(sys.argv[5]) if len(sys.argv) > 5 else 10
inv = {v: k for k, v in XMAP.items()}
print("col " + " ".join(f"   s{s}(row)   " for s in shifts))
for k in range(k_cols):
    cells = []
    for j, s in enumerate(shifts):
        sp = spans(runs[big[first - 1 + j]], freq)
        row = (k + s) % rows
        c = (inv[k], row + 2)
        e = sorted(sp.get((c, "writer_br0_early"), []), key=lambda v: v[1])
        w = sorted(sp.get((c, "writer_br0_read_wait"), []), key=lambda v: v[1])
        if e and w:
            cells.append(f"{w[0][2] - e[0][2]:4.2f}({row})@{w[0][2]:4.2f}")
        else:
            cells.append("      -      ")
    print(f"{k:3d} " + " ".join(cells))
