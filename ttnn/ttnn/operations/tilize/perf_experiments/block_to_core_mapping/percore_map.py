#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""PER-CORE tail diagnosis for tilize (idea `block_to_core_mapping`, step (a)).

`zone_report.py` aggregates over cores (mean / max only). This prints the SAME
`*-KERNEL` spans as an 8x8 GRID MAP, plus the absolute start and end of each
core's kernel relative to the run's earliest start, so the three candidate
causes of the 1.24-1.43x max/mean tail can be told apart:

  * DURATION spread  (core does the same work slower -> contention / routing)
  * START spread     (dispatch go-signal skew -> nothing a permutation can fix)
  * END spread       (what the wall actually is)

Physical NoC coordinates in the log are ranked to logical grid coordinates (the
worker rows/columns are not contiguous on wormhole_b0), so the map is in the
same coordinate system as the plan's `assignment`.

    python3 .../block_to_core_mapping/percore_map.py --log <csv> [--run N] \
        [--plan logs/plan_baseline_focus.json]
"""
import argparse
import collections
import csv
import json
import os
import statistics
import sys

DEFAULT_LOG = os.environ.get("TILIZE_PROFILE_LOG", "generated/profiler/.logs/profile_log_device.csv")


def load(path):
    rows = list(csv.reader(open(path)))
    return [r for r in rows[2:] if len(r) >= 12]


def collect(path):
    """(run, (px,py), risc) -> (min_start, max_end) over the *-KERNEL zones."""
    st = collections.defaultdict(lambda: [None, None])
    markers = collections.Counter()
    for r in load(path):
        core = (int(r[1]), int(r[2]))
        risc = r[3].strip()
        t = int(r[5])
        run = int(r[7])
        zone = r[10].strip()
        typ = r[11].strip()
        markers[(run, core, risc)] += 1
        if not zone.endswith("-KERNEL"):
            continue
        cur = st[(run, core, risc)]
        if typ == "ZONE_START":
            cur[0] = t if cur[0] is None else min(cur[0], t)
        elif typ == "ZONE_END":
            cur[1] = t if cur[1] is None else max(cur[1], t)
    return st, markers


def grid_maps(st, run, risc):
    sel = {c: v for (rn, c, rk), v in st.items() if rn == run and rk == risc and None not in v}
    if not sel:
        return None
    xs = sorted({c[0] for c in sel})
    ys = sorted({c[1] for c in sel})
    t0 = min(v[0] for v in sel.values())
    out = {}
    for (px, py), (s, e) in sel.items():
        out[(xs.index(px), ys.index(py))] = (e - s, s - t0, e - t0)
    return out, xs, ys


def print_map(title, cells, key, width=8):
    print(f"\n  {title}")
    ys = sorted({y for _, y in cells})
    xs = sorted({x for x, _ in cells})
    print("      " + "".join(f"{x:>8d}" for x in xs))
    for y in ys:
        row = "".join(f"{cells[(x, y)][key]:8.0f}" if (x, y) in cells else "       ." for x in xs)
        print(f"  y={y}  {row}")


def corr(pairs):
    """Pearson r over [(a, b), ...]."""
    if len(pairs) < 3:
        return float("nan")
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    if va == 0 or vb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--plan", default=None, help="logs/plan_<mode>_<cases>.json (adds block-id correlations)")
    ap.add_argument("--riscs", default="NCRISC,BRISC,TRISC_0,TRISC_1,TRISC_2")
    args = ap.parse_args()

    st, markers = collect(args.log)
    runs = sorted({k[0] for k in st})
    block_of = None
    if args.plan:
        recs = json.loads(open(args.plan).read())
        block_of = {(r["label"]): {(a[0], a[1]): a[2] for a in r["assignment"]} for r in recs}

    for run in runs:
        if args.run is not None and run != args.run:
            continue
        print(f"\n================ run host ID {run} ================")
        for risc in args.riscs.split(","):
            got = grid_maps(st, run, risc)
            if not got:
                continue
            cells, xs, ys = got
            durs = [v[0] for v in cells.values()]
            ends = [v[2] for v in cells.values()]
            starts = [v[1] for v in cells.values()]
            print(
                f"\n--- {risc}: {len(cells)} cores | duration mean {statistics.mean(durs):.0f} "
                f"max {max(durs):.0f} ({max(durs)/max(1,statistics.mean(durs)):.2f}x) | "
                f"start spread {max(starts)-min(starts):.0f} | end spread {max(ends)-min(ends):.0f} "
                f"| wall(end max) {max(ends):.0f}"
            )
            print_map(f"{risc} DURATION ns", cells, 0)
            print_map(f"{risc} START ns (rel)", cells, 1)
            print_map(f"{risc} END ns (rel)", cells, 2)
            # correlations
            rows = [(y, cells[(x, y)][0]) for (x, y) in cells]
            cols = [(x, cells[(x, y)][0]) for (x, y) in cells]
            print(f"    corr(duration, grid_row) = {corr(rows):+.2f}   corr(duration, grid_col) = {corr(cols):+.2f}")
            print(f"    corr(duration, start)    = {corr([(cells[c][1], cells[c][0]) for c in cells]):+.2f}")
            if block_of:
                for label, m in block_of.items():
                    pairs = [(m[c], cells[c][0]) for c in cells if c in m]
                    if not pairs:
                        continue
                    print(
                        f"    [{label}] corr(duration, block_id) = {corr(pairs):+.2f}  "
                        f"corr(duration, block_id%12) = {corr([(b % 12, d) for b, d in pairs]):+.2f}  "
                        f"corr(duration, (8*block_id)%12) = {corr([((8 * b) % 12, d) for b, d in pairs]):+.2f}"
                    )
                    by = collections.defaultdict(list)
                    for b, d in pairs:
                        by[(8 * b) % 12].append(d)
                    print(
                        "      duration by first-write bank (8b%12): "
                        + "  ".join(f"{k}:{statistics.mean(v):.0f}" for k, v in sorted(by.items()))
                    )
        worst = max(((k, c) for k, c in markers.items() if k[0] == run), key=lambda kc: kc[1], default=None)
        if worst:
            print(f"\n  marker budget: worst (core,RISC) = {worst[0][1]} {worst[0][2]} at {worst[1]}/250")


if __name__ == "__main__":
    sys.exit(main())
