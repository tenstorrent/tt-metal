#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""WHY the skew is null: how each grid ROW's kernel span responds to the WORK it
is given.

For every mode in a run, each grid row y was handed a known block width w[y].
Regressing the row's mean `*-KERNEL` span against w[y] across the modes gives

    duration[y] ~ a[y] + b[y] * w[y]

  * `b[y]` is the row's MARGINAL cost of one more tile of block width — what an
    extra tile actually costs to serve on that row;
  * `a[y]` is the part that does NOT shrink when the row is given less work —
    the queueing/wait component.

M2 needs a[y] to be small: only then does taking work off a straggler make it
finish proportionally earlier. M1 predicts a large a[y] on exactly the slow rows
(they are WAITING, not working), so the redistribution trade is priced
`b[fast] > b[slow]` — you pay more to add a tile to a fast row than you save by
removing one from a slow row, and the wall gets worse.
"""
import argparse
import collections
import csv
import json
import statistics
import sys


def _ladders():
    """`bench.LADDERS`, read with `ast` rather than imported — importing bench
    pulls in ttnn, which this pure-CSV analysis has no need for."""
    import ast
    import os

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "LADDERS":
            return ast.literal_eval(node.value)
    raise AssertionError("LADDERS not found in bench.py")


LADDERS = _ladders()


def spans(path):
    """(run, risc, logical (x,y)) -> duration ns, over the *-KERNEL zones."""
    rows = list(csv.reader(open(path)))[2:]
    st = collections.defaultdict(lambda: [None, None])
    for r in rows:
        if len(r) < 12 or not r[10].strip().endswith("-KERNEL"):
            continue
        k = (int(r[7]), r[3].strip(), int(r[1]), int(r[2]))
        t = int(r[5])
        cur = st[k]
        if r[11].strip() == "ZONE_START":
            cur[0] = t if cur[0] is None else min(cur[0], t)
        else:
            cur[1] = t if cur[1] is None else max(cur[1], t)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--ops", required=True)
    ap.add_argument("--dispatches", required=True)
    a = ap.parse_args()

    opsrows = [r for r in csv.DictReader(open(a.ops)) if r.get("OP CODE", "").strip()]
    disp = json.loads(open(a.dispatches).read())["dispatches"]
    run_of = {}
    for r, d in zip(opsrows, disp):
        if d["phase"] == "measure":
            run_of.setdefault(d["mode"], []).append(int(r["GLOBAL CALL COUNT"]))

    st = spans(a.dev)
    pxs = sorted({k[2] for k in st})
    pys = sorted({k[3] for k in st})

    for risc in ("NCRISC", "BRISC"):
        print(f"\n===== {risc}: row mean span (ns) per mode, and the response to width =====")
        # per mode -> per logical row -> mean over its cores, median over reps
        table = {}
        for mode, runs in run_of.items():
            per_row = collections.defaultdict(list)
            for run in runs:
                acc = collections.defaultdict(list)
                for (rn, rk, px, py), v in st.items():
                    if rn != run or rk != risc or None in v:
                        continue
                    acc[pys.index(py)].append(v[1] - v[0])
                for y, ds in acc.items():
                    per_row[y].append(statistics.mean(ds))
            table[mode] = {y: statistics.median(v) for y, v in per_row.items()}

        modes = [m for m in run_of if m in LADDERS or m == "baseline"]
        print(f"{'mode':16s} " + "".join(f"  y={y}" + " " * 4 for y in range(8)))
        for m in modes:
            print(f"{m:16s} " + "".join(f"{table[m].get(y, 0):8.0f}" for y in range(8)))
        print(f"{'width w[y]':16s}")
        for m in modes:
            lad = LADDERS.get(m, (8,) * 8) if m != "baseline" else (8,) * 8
            print(f"  {m:14s} " + "".join(f"{w:8d}" for w in lad))

        print(f"\n  per-row least squares  duration ~ a + b*w   (a = the part work does NOT buy back)")
        print(f"  {'row':>4s} {'a (ns)':>9s} {'b (ns/tile)':>12s} {'a share at w=8':>15s}")
        for y in range(8):
            pts = []
            for m in modes:
                lad = LADDERS.get(m, (8,) * 8) if m != "baseline" else (8,) * 8
                if y in table[m]:
                    pts.append((lad[y], table[m][y]))
            if len({p[0] for p in pts}) < 2:
                continue
            n = len(pts)
            sw = sum(p[0] for p in pts)
            sd = sum(p[1] for p in pts)
            swd = sum(p[0] * p[1] for p in pts)
            sww = sum(p[0] * p[0] for p in pts)
            b = (n * swd - sw * sd) / (n * sww - sw * sw)
            aa = (sd - b * sw) / n
            print(f"  {y:>4d} {aa:9.0f} {b:12.0f} {100*aa/max(1.0,aa+8*b):14.0f}%")


if __name__ == "__main__":
    sys.exit(main())
