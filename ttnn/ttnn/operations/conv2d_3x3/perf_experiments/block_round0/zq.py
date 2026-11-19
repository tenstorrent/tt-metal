# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: zone instances on chosen cores of one dispatch.
    python3 zq.py <log> <run#> <zone[,zone...]> [x,y ...]   (no cores: every core that has the zone)"""
import sys

from r0view import XMAP, load, spans

freq, runs = load(sys.argv[1])
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
sp = spans(runs[big[int(sys.argv[2]) - 1]], freq)
zones = sys.argv[3].split(",")
want = [tuple(int(v) for v in a.split(",")) for a in sys.argv[4:]]
cores = sorted({c for c, _ in sp}, key=lambda c: (XMAP[c[0]], c[1])) if not want else want
for c in cores:
    for z in zones:
        v = sorted(sp.get((c, z), []), key=lambda e: e[1])
        if v:
            print(
                f"{str(c):>8} {str((XMAP[c[0]], c[1] - 2)):>8} {z:<28} "
                + " ".join(f"{a:5.2f}-{b:5.2f}" for _, a, b in v[:6])
            )
