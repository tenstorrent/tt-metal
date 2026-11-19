# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-round end of a repeated zone (default writer_weight_fetch) over the cores of one dispatch: p50 / max per round,
plus the walk end (compute_matmul_walk on TRISC_2) and the kernel end. Host-only.
    python3 rounds.py <profile_log_device.csv[.gz]> <run#> [zone]"""
import collections
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from zones import load, spans  # noqa: E402

freq, runs = load(sys.argv[1])
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
zone = sys.argv[3] if len(sys.argv) > 3 else "writer_weight_fetch"
for idx in sys.argv[2].split(","):
    sp = spans(runs[big[int(idx) - 1]], freq)
    per = collections.defaultdict(list)
    walk, kend = {}, collections.defaultdict(float)
    for c, rk, z, s, e in sp:
        if z == zone:
            per[c].append(e)
        if z == "compute_matmul_walk" and rk == "TRISC_2":
            walk[c] = e
        if z.endswith("-KERNEL"):
            kend[c] = max(kend[c], e)
    n = max(len(v) for v in per.values())
    out = []
    for r in range(n):
        ends = [sorted(v)[r] for v in per.values() if len(v) > r]
        out.append(f"r{r} {statistics.median(ends):5.2f}/{max(ends):5.2f}")
    print(f"run {idx}: " + "  ".join(out))
    print(
        f"        walk end p50 {statistics.median(walk.values()):5.2f} max {max(walk.values()):5.2f}   kernel end p50 "
        f"{statistics.median(kend.values()):5.2f} max {max(kend.values()):5.2f}"
    )
