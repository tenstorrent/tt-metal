# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage zone report for rms_norm, off `generated/profiler/.logs/profile_log_device.csv`.

Folds the raw ZONE_START / ZONE_END marker stream into per-(zone, RISC) totals for
one dispatch, and runs the two integrity checks that the marker budget makes
mandatory (see kernel_lib/perf_instrumentation.hpp):

  * marker count per (core, RISC) against PROFILER_L1_OPTIONAL_MARKER_COUNT (250) —
    exhaustion is SILENT, so a RISC at the cap means the report is partial;
  * the last user zone's end against that RISC's *-KERNEL span — zones that stop
    well short of the span invent a dominant stage and look complete doing it.

Usage:
    python3 -m ttnn.operations.rms_norm._zone_report [<profile_log_device.csv>] [--run <host_id>]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict

DEFAULT_LOG = "generated/profiler/.logs/profile_log_device.csv"
MARKER_CAP = 250
FW_ZONES = ("BRISC-FW", "NCRISC-FW", "TRISC-FW", "BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL")


def load(path):
    with open(path) as fh:
        header = fh.readline()
        mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader(fh, skipinitialspace=True))
    return mhz, rows


def report(path=DEFAULT_LOG, run=None):
    mhz, rows = load(path)
    runs = sorted({r["run host ID"] for r in rows if r["run host ID"]}, key=lambda v: int(v))
    run = run or runs[-1]
    rows = [r for r in rows if r["run host ID"] == run]

    # marker budget per (core, risc)
    markers = defaultdict(int)
    for r in rows:
        markers[(r["core_x"], r["core_y"], r["RISC processor type"])] += 1

    open_stack = defaultdict(list)
    tot = defaultdict(float)
    cnt = defaultdict(int)
    span = {}
    last_user_end = defaultdict(float)
    for r in rows:
        key = (r["core_x"], r["core_y"], r["RISC processor type"])
        name = r["zone name"]
        cyc = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            open_stack[(key, name)].append(cyc)
        elif r["type"] == "ZONE_END":
            st = open_stack[(key, name)]
            if not st:
                continue
            dur = (cyc - st.pop()) / mhz * 1000.0  # ns
            zk = (name, r["RISC processor type"])
            tot[zk] += dur
            cnt[zk] += 1
            if name.endswith("-KERNEL"):
                span[key] = (st[0] if st else 0, dur)
            elif name not in FW_ZONES:
                last_user_end[key] = max(last_user_end[key], cyc)

    print(f"# run host ID {run}   clock {mhz:.0f} MHz   cores*riscs {len(markers)}")
    print(f"{'zone':<22} {'risc':<8} {'execs':>6} {'total ns':>12} {'ns/exec':>10}")
    for (name, risc), v in sorted(tot.items(), key=lambda kv: -kv[1]):
        if name in FW_ZONES:
            continue
        print(f"{name:<22} {risc:<8} {cnt[(name, risc)]:>6} {v:>12.0f} {v / cnt[(name, risc)]:>10.0f}")
    print("\n# kernel spans (per core/RISC) and zone coverage")
    for key, (_, dur) in sorted(span.items()):
        print(
            f"  core({key[0]},{key[1]}) {key[2]:<8} kernel span {dur:>10.0f} ns  markers {markers[key]:>4}"
            f"{'  <-- AT MARKER CAP, REPORT IS PARTIAL' if markers[key] >= MARKER_CAP else ''}"
        )
    over = [k for k, v in markers.items() if v >= MARKER_CAP]
    print(f"\n# marker-cap check: {len(over)} of {len(markers)} (core,RISC) at/above {MARKER_CAP}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    run = None
    if "--run" in args:
        i = args.index("--run")
        run = args[i + 1]
        args = args[:i] + args[i + 2 :]
    report(args[0] if args else DEFAULT_LOG, run)
