#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage device-zone report for tilize (Perf 1 breakdown tool).

Reads `generated/profiler/.logs/profile_log_device.csv` (written by any
`run_safe_pytest.sh --profile` run) and prints, per dispatch (`run host ID`):

  * the `*-KERNEL` span per RISC — the denominator every stage is a share OF;
  * every user zone's per-core mean / max total, and its share of that RISC's
    kernel span;
  * a MARKER-BUDGET check per (core, RISC). Budget exhaustion is SILENT (the
    profiler simply stops recording and the report still looks complete), so a
    breakdown must not be ranked until `coverage` is ~100% and the marker count
    is clear of the 250 cap. See .claude/references/device-zone-scope-attribution.md.

    python3 ttnn/ttnn/operations/tilize/perf_experiments/zone_report.py [--run N]
"""
import argparse
import collections
import csv
import os
import sys

LOG = os.environ.get("TILIZE_PROFILE_LOG", "generated/profiler/.logs/profile_log_device.csv")
FREQ_MHZ = 1000.0  # wormhole_b0; the log's own header states it


def load(path):
    rows = list(csv.reader(open(path)))
    return [r for r in rows[2:] if len(r) >= 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=None, help="run host ID (default: all)")
    ap.add_argument("--log", default=LOG)
    args = ap.parse_args()

    data = load(args.log)
    # (run, core, risc, zone) -> list of (start, end)
    open_stack = collections.defaultdict(list)
    spans = collections.defaultdict(list)
    markers = collections.Counter()
    for r in data:
        core = (r[1].strip(), r[2].strip())
        risc = r[3].strip()
        t = int(r[5])
        run = int(r[7])
        zone = r[10].strip()
        typ = r[11].strip()
        key = (run, core, risc, zone)
        markers[(run, core, risc)] += 1
        if typ == "ZONE_START":
            open_stack[key].append(t)
        elif typ == "ZONE_END" and open_stack[key]:
            spans[key].append((open_stack[key].pop(), t))

    runs = sorted({k[0] for k in spans})
    for run in runs:
        if args.run is not None and run != args.run:
            continue
        # kernel span per risc = max(end) - min(start) over the *-KERNEL zones
        kspan = {}
        for (rn, core, risc, zone), sp in spans.items():
            if rn != run or not zone.endswith("-KERNEL"):
                continue
            kspan.setdefault(risc, []).append(max(e for _, e in sp) - min(s for s, _ in sp))
        print(f"\n=== run host ID {run} ===")
        for risc in sorted(kspan):
            v = kspan[risc]
            print(f"  {risc:8s} KERNEL span  mean {sum(v)/len(v):9.0f} ns   max {max(v):9.0f} ns  ({len(v)} cores)")
        agg = collections.defaultdict(list)
        cov = collections.defaultdict(list)
        for (rn, core, risc, zone), sp in spans.items():
            if rn != run or zone.endswith("-KERNEL") or zone.endswith("-FW"):
                continue
            agg[(risc, zone)].append(sum(e - s for s, e in sp))
            cov[(risc, core)].append(max(e for _, e in sp))
        print(f"  {'RISC':8s} {'zone':24s} {'mean ns':>9s} {'max ns':>9s} {'%span':>7s} {'execs':>6s}")
        for (risc, zone), v in sorted(agg.items()):
            span = sum(kspan.get(risc, [0])) / max(1, len(kspan.get(risc, [1])))
            n = len(spans[(run, ("1", "1"), risc, zone)]) if (run, ("1", "1"), risc, zone) in spans else 0
            print(
                f"  {risc:8s} {zone:24s} {sum(v)/len(v):9.0f} {max(v):9.0f} "
                f"{100*(sum(v)/len(v))/max(span,1):6.1f}% {n:6d}"
            )
        worst = max(((k, c) for k, c in markers.items() if k[0] == run), key=lambda kc: kc[1], default=None)
        if worst:
            print(f"  marker budget: worst (core,RISC) = {worst[0][1]} {worst[0][2]} at {worst[1]}/250 markers")


if __name__ == "__main__":
    sys.exit(main())
