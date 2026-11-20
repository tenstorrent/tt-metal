"""Column fill resource model (aspp-like BLOCK K split, grid rows only): per column ring of 12 physical rows, the bytes on
every NoC0 (+y) / NoC1 (-y) link, every owner's injection and every reader's ejection per NoC, for a NoC assignment of
each (core, owner shard) piece. Reports the binding resource in us at 62 B/cycle (BH measured one-stream L1 read rate,
noc_latencies.yaml) and 1350 MHz, for: base (higher shards on NoC1 except on the column sender row), all rows split,
and the LP optimum (rows of any piece on either NoC, long way allowed).

    python3 noc_lp.py [case] [sender_row]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import linkload  # noqa: E402

RATE = 62 * 1.35e3  # bytes per us


def pieces(case):
    H, W, cin, cinp, d, shard_rows, sizes, tb = linkload.CASES[case]
    g = linkload.geom(H, W, cin, cinp, d)
    out = []  # (m, s, rows)
    for m, (start, n) in enumerate(linkload.ranges_of(sizes)):
        rows = [r for f, c in linkload.pd._host_fill_segments(g, start, n) for r in range(f, f + c)]
        by = {}
        for r in rows:
            by[r // shard_rows] = by.get(r // shard_rows, 0) + 1
        out += [(m, s, c) for s, c in sorted(by.items())]
    return out, (cinp // 32) * tb, len(sizes)


def loads(pcs, x1, row_bytes):
    """x1[i]: rows of piece i on NoC1. Returns dict of resource -> bytes."""
    res = {}

    def add(k, v):
        res[k] = res.get(k, 0) + v

    for (m, s, c), r1 in zip(pcs, x1):
        ys, ym = 2 + s, 2 + m
        for noc, rows in ((0, c - r1), (1, r1)):
            if rows <= 0:
                continue
            b = rows * row_bytes
            add(("inj", noc, s), b)
            add(("ej", noc, m), b)
            lk = linkload.links0(ys, ym) if noc == 0 else linkload.links1(ys, ym)
            for y in lk:
                add(("link", noc, y), b)
    return res


def report(name, pcs, x1, row_bytes):
    res = loads(pcs, x1, row_bytes)
    top = sorted(res.items(), key=lambda kv: -kv[1])[:4]
    print(f"  {name:10s} bound {top[0][1] / RATE:5.2f} us  top: " + ", ".join(f"{k}={v / 1024:.0f}KB" for k, v in top))
    return res


def lp(pcs, row_bytes, fixed=None):
    n = len(pcs)
    keys = sorted(
        {k for i in range(n) for k in loads([pcs[i]], [0], row_bytes)}
        | {k for i in range(n) for k in loads([pcs[i]], [pcs[i][2]], row_bytes)}
    )
    kix = {k: j for j, k in enumerate(keys)}
    # resource load = base (all on NoC0) + sum_i x_i * (coef1_i - coef0_i) per row; T is var n.
    A = np.zeros((len(keys), n + 1))
    b = np.zeros(len(keys))
    for i, p in enumerate(pcs):
        l0 = loads([p], [0], row_bytes)
        l1 = loads([p], [p[2]], row_bytes)
        for k, v in l0.items():
            b[kix[k]] -= v  # move to rhs
            A[kix[k], i] -= v / p[2]
        for k, v in l1.items():
            A[kix[k], i] += v / p[2]
    A[:, n] = -1.0
    c = np.zeros(n + 1)
    c[n] = 1.0
    bounds = [(0, p[2]) for p in pcs] + [(0, None)]
    r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return r.x[:n]


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "aspp1"
    sender = int(sys.argv[2]) if len(sys.argv) > 2 else None
    pcs, row_bytes, nrows = pieces(case)
    print(f"{case}: {len(pcs)} pieces, sender row {sender}")
    base = [c if (s > m and m != sender) else 0 for m, s, c in pcs]
    report("base", pcs, base, row_bytes)
    allsplit = [c if s > m else 0 for m, s, c in pcs]
    report("all_split", pcs, allsplit, row_bytes)
    x = lp(pcs, row_bytes)
    report("lp", pcs, x, row_bytes)
    for (m, s, c), r1 in zip(pcs, x):
        if r1 > 0.01 and not (s > m and abs(r1 - c) < 0.01):
            print(f"    lp: m{m} s{s} rows {c}: {r1:.2f} on NoC1 (h1 {(2 + s - 2 - m) % 12})")


if __name__ == "__main__":
    main()
