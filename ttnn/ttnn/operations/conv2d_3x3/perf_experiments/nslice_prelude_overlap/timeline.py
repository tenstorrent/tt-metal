# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One core's zone timeline of one run (no ttnn): us from the run's earliest marker over all cores.

    python3 timeline.py <profile_log_device.csv> [--run RUN_ID] [--core X,Y] [--risc NCRISC,TRISC_0] [--agg]

--agg: per zone, per RISC, the core's summed us and count (all runs listed when --run is omitted).
"""

import argparse
import collections
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("zone_report", Path(__file__).parent.parent / "zone_report.py")
zr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zr)


def spans_of(ev, freq):
    t0 = min(e[2] for e in ev)
    stacks = {}
    spans = collections.defaultdict(list)
    for core, risc, t, zone, kind in sorted(ev, key=lambda e: e[2]):
        key = (core, risc, zone)
        if kind == "ZONE_START":
            stacks.setdefault(key, []).append(t)
        elif kind == "ZONE_END" and stacks.get(key):
            s = stacks[key].pop()
            spans[core].append(((s - t0) / freq, (t - t0) / freq, risc, zone))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--core", default="1,1")
    ap.add_argument("--risc", default="NCRISC,TRISC_0")
    ap.add_argument("--agg", action="store_true")
    args = ap.parse_args()
    freq, runs = zr.load(args.log)
    core = tuple(int(v) for v in args.core.split(","))
    riscs = set(args.risc.split(","))
    for run_id in sorted(runs):
        if args.run is not None and run_id != args.run:
            continue
        sp = sorted(s for s in spans_of(runs[run_id], freq)[core] if s[2] in riscs)
        print(f"=== run {run_id} core {core}")
        if args.agg:
            agg = collections.defaultdict(lambda: [0.0, 0])
            for s, e, risc, zone in sp:
                agg[(risc, zone)][0] += e - s
                agg[(risc, zone)][1] += 1
            for (risc, zone), (tot, n) in sorted(agg.items()):
                print(f"  {risc:<8} {zone:<32} {tot:8.2f} us  x{n}")
            continue
        for s, e, risc, zone in sp:
            print(f"  {s:8.2f} {e:8.2f} {e - s:7.2f}  {risc:<8} {zone}")


if __name__ == "__main__":
    main()
