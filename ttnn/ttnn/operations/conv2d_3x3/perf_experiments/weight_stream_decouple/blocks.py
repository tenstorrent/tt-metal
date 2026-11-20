# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: per-core M-block timeline (compute walk / bias, reader fill, writer store) of one program run.

    python3 blocks.py <dev.csv[.gz]> <run ordinal among >=100-core runs> [--cores N]

Prints, for N cores spread over the work order, each M-block's walk start / end, bias end, the reader fill window and
the writer store window (us from the run's first marker), then per-zone medians over all cores of the boundary pieces:
fill duration, store duration, bias duration, gap walk_end -> next walk start.
"""

import statistics
import sys

from rounds import load, zones


def main():
    path, ordinal = sys.argv[1], int(sys.argv[2])
    ncores = int(sys.argv[sys.argv.index("--cores") + 1]) if "--cores" in sys.argv else 4
    freq, runs = load(path)
    big = [rid for rid in sorted(runs) if len({e[0] for e in runs[rid]}) >= 100]
    rid = big[ordinal]
    zs = zones(runs[rid])
    t0 = min(e[2] for e in runs[rid])
    us = lambda c: (c - t0) / freq
    d = lambda se: (se[1] - se[0]) / freq
    cores = sorted({k[0] for k in zs})
    stats = {"fill": [], "store": [], "bias": [], "gap": [], "walk": []}
    pick = set(cores[:: max(1, len(cores) // ncores)][:ncores])
    for c in cores:
        walk = zs[(c, "TRISC_0")].get("compute_matmul_walk", [])
        bias = zs[(c, "TRISC_0")].get("compute_bias_add", [])
        fill = zs[(c, "NCRISC")].get("reader_fill", [])
        store = zs[(c, "BRISC")].get("writer_store", [])
        stats["fill"] += [d(x) for x in fill[1:]]
        stats["store"] += [d(x) for x in store]
        stats["bias"] += [d(x) for x in bias]
        stats["walk"] += [d(x) for x in walk]
        stats["gap"] += [(walk[i + 1][0] - walk[i][1]) / freq for i in range(len(walk) - 1)]
        if c in pick:
            print(f"core {c}:")
            for m in range(len(walk)):
                f = fill[m] if m < len(fill) else (t0, t0)
                st = store[m] if m < len(store) else (t0, t0)
                b = bias[m] if m < len(bias) else (t0, t0)
                print(
                    f"  m{m:2d} walk {us(walk[m][0]):8.1f}-{us(walk[m][1]):8.1f} ({d(walk[m]):6.1f})"
                    f" bias ->{us(b[1]):8.1f} ({d(b):5.1f}) fill {us(f[0]):8.1f}-{us(f[1]):8.1f} ({d(f):5.1f})"
                    f" store {us(st[0]):8.1f}-{us(st[1]):8.1f} ({d(st):5.1f})"
                )
    for k, v in stats.items():
        if v:
            print(f"{k:6s} p50 {statistics.median(v):7.1f}  p90 {sorted(v)[int(0.9 * len(v))]:7.1f}  n {len(v)}")


if __name__ == "__main__":
    main()
