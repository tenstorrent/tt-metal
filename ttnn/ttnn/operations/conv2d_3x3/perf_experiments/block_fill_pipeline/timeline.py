# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-core phase timeline of one profiled dispatch (no device, no ttnn).

    python3 timeline.py <profile_log_device.csv[.gz]> [--runs i,j] [--cores x,y;x,y] [--zones a,b]

For each run (dispatch, in order): every zone's first start and last end per core, in us from the run's earliest
marker over all cores; then the latest-ending cores' phase ends and the distribution (p50 / max) of each zone's
first start and last end over cores. Shows which phase moved when a variant changes the op time.
"""

import argparse
import collections
import gzip
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import zone_report  # noqa: E402

KEY = [
    "reader_fill",
    "compute_untilize",
    "reader_gather",
    "compute_tilize",
    "compute_matmul_walk",
    "compute_combine",
    "writer_slab_dead_wait",
    "writer_fill",
]


def load(path):
    if str(path).endswith(".gz"):
        tmp = Path("/tmp") / (Path(path).name[:-3] + ".bfp")
        with gzip.open(path, "rb") as f, open(tmp, "wb") as g:
            g.write(f.read())
        path = tmp
    return zone_report.load(path)


def spans(events, freq):
    t0 = min(e[2] for e in events)
    first = {}
    last = {}
    stacks = collections.defaultdict(list)
    kernel_end = collections.defaultdict(int)
    for core, risc, t, zone, kind in sorted(events, key=lambda e: e[2]):
        kernel_end[core] = max(kernel_end[core], t)
        key = (core, risc, zone)
        if kind == "ZONE_START":
            stacks[key].append(t)
        elif kind == "ZONE_END" and stacks[key]:
            s = stacks[key].pop()
            k = (core, zone if risc != "TRISC_0" or not zone.startswith("compute") else zone)
            if risc in ("TRISC_1", "TRISC_2") and zone.startswith("compute"):
                continue
            first[k] = min(first.get(k, s), s)
            last[k] = max(last.get(k, t), t)
    us = lambda c: (c - t0) / freq  # noqa: E731
    return (
        {k: us(v) for k, v in first.items()},
        {k: us(v) for k, v in last.items()},
        {c: us(v) for c, v in kernel_end.items()},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--zones", default=",".join(KEY))
    ap.add_argument("--worst", type=int, default=3)
    args = ap.parse_args()
    freq, runs = load(args.log)
    ids = sorted(rid for rid, ev in runs.items() if len({c for c, *_ in ev}) >= 8)
    sel = range(len(ids)) if args.runs is None else [int(x) for x in args.runs.split(",")]
    zones = args.zones.split(",")
    for i in sel:
        first, last, kend = spans(runs[ids[i]], freq)
        cores = sorted(kend)
        print(
            f"=== dispatch {i} (run {ids[i]}): end max {max(kend.values()):.2f} us p50 {statistics.median(kend.values()):.2f}"
        )
        print(f"  {'zone':<26} {'start p50':>9} {'start max':>9} {'end p50':>9} {'end max':>9}")
        for z in zones:
            s = [first[(c, z)] for c in cores if (c, z) in first]
            e = [last[(c, z)] for c in cores if (c, z) in last]
            if not s:
                continue
            print(f"  {z:<26} {statistics.median(s):9.2f} {max(s):9.2f} {statistics.median(e):9.2f} {max(e):9.2f}")
        worst = sorted(cores, key=lambda c: -kend[c])[: args.worst]
        for c in worst:
            parts = []
            for z in zones:
                if (c, z) in first:
                    parts.append(f"{z.split('_', 1)[1]} {first[(c, z)]:.1f}-{last[(c, z)]:.1f}")
            print(f"  core {c} end {kend[c]:.2f}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
