# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: per column, the round-0 reader's read (early issue end -> landed) and multicast zones, given Q rows.
    python3 qtab.py <log> <run#> <q rows comma list | s (the column senders)> [shift]"""
import sys

from r0view import XMAP, load, spans

freq, runs = load(sys.argv[1])
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
sp = spans(runs[big[int(sys.argv[2]) - 1]], freq)
inv = {v: k for k, v in XMAP.items()}
rows = [k for k in range(8)] if sys.argv[3] == "s" else [int(v) for v in sys.argv[3].split(",")]
for k, r in enumerate(rows):
    c = (inv[k], r + 2)
    g = lambda z: sorted(sp.get((c, z), []), key=lambda v: v[1])
    e, w, m, d = g("writer_br0_early"), g("writer_br0_read_wait"), g("writer_br0_mcast"), g("writer_br0_dual")
    f = lambda v: f"{v[0][1]:5.2f}-{v[0][2]:5.2f}" if v else "     -     "
    rd = f"{w[0][2] - e[0][2]:4.2f}" if e and w else "  - "
    print(f"col {k} row {r} {c}: early {f(e)} wait {f(w)} read {rd} mcast {f(m)} dual {f(d)}")
