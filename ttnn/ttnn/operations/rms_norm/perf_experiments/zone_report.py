# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage device-zone report for one profiled run of rms_norm.

Reads generated/profiler/.logs/profile_log_device.csv (written by a `scripts/run_safe_pytest.sh
--profile` run), pairs ZONE_START / ZONE_END rows per (core, RISC, zone) and prints, per RISC and
zone: executions per core, median / max ns across cores, and the number of cores that recorded it.
Also prints the *-KERNEL span max / p50 (core balance) and the marker count per (core, RISC) so a
truncated profile (marker budget exhausted) is visible.

    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/zone_report.py [--run-id ID] [--csv PATH]

Host-only: reads a CSV, never touches the device.
"""

from __future__ import annotations

import argparse
import os
import statistics
from collections import defaultdict

RISC_ORDER = ["NCRISC", "BRISC", "TRISC_0", "TRISC_1", "TRISC_2"]
MARKER_CAP = 250  # PROFILER_L1_OPTIONAL_MARKER_COUNT per RISC


def read_rows(path):
    with open(path) as f:
        lines = f.read().splitlines()
    freq_mhz = 1000.0
    for part in lines[0].split(","):
        if "CHIP_FREQ" in part:
            freq_mhz = float(part.split(":")[1])
    rows = [[x.strip() for x in ln.split(",")] for ln in lines[2:] if ln.strip()]
    return [r for r in rows if len(r) >= 12], freq_mhz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv"),
    )
    ap.add_argument("--run-id", default=None, help="run host ID to select (default: the largest present)")
    ap.add_argument("--zone-filter", default=None, help="substring; only zones containing it")
    args = ap.parse_args()

    rows, freq_mhz = read_rows(args.csv)
    ns_per_cycle = 1000.0 / freq_mhz
    run_ids = sorted({int(r[7]) for r in rows if r[7].isdigit()})
    run_id = int(args.run_id) if args.run_id else run_ids[-1]
    rows = [r for r in rows if r[7].isdigit() and int(r[7]) == run_id]
    print(f"run host id {run_id} (present: {run_ids[-5:]}), CHIP_FREQ {freq_mhz:.0f} MHz, {len(rows)} marker rows")

    # (core, risc, zone) -> [starts], [ends]
    starts, ends = defaultdict(list), defaultdict(list)
    markers = defaultdict(int)
    for r in rows:
        core = (int(r[1]), int(r[2]))
        risc, cyc, zone, typ = r[3], int(r[5]), r[10], r[11]
        markers[(core, risc)] += 1
        key = (core, risc, zone)
        if typ == "ZONE_START":
            starts[key].append(cyc)
        elif typ == "ZONE_END":
            ends[key].append(cyc)

    # durations per key
    durs = {}
    for key, s in starts.items():
        e = sorted(ends.get(key, []))
        s = sorted(s)
        d = [(ee - ss) * ns_per_cycle for ss, ee in zip(s, e)]
        if d:
            durs[key] = d

    # marker budget check
    hot = [(k, n) for k, n in markers.items() if n >= MARKER_CAP - 4]
    if hot:
        print(
            f"WARNING: {len(hot)} (core,RISC) at the marker cap ({MARKER_CAP}) -> profile TRUNCATED for them: {hot[:6]}"
        )

    # kernel span balance
    print("\n== kernel span (per RISC): max / p50 across cores ==")
    for risc in RISC_ORDER:
        spans = [sum(d) for (c, rr, z), d in durs.items() if rr == risc and z.endswith("-KERNEL")]
        if spans:
            p50 = statistics.median(spans)
            print(
                f"  {risc:8s} cores={len(spans):3d} max={max(spans):9.0f} p50={p50:9.0f} ratio={max(spans)/p50 if p50 else 0:.2f}"
            )

    # per-zone table
    print("\n== zones: per RISC, per zone (ns summed over executions on a core; stats across cores) ==")
    print(
        f"  {'RISC':8s} {'zone':32s} {'cores':>5s} {'exec/core':>9s} {'median_ns':>10s} {'max_ns':>10s} {'min_ns':>10s}"
    )
    by_rz = defaultdict(list)
    for (core, risc, zone), d in durs.items():
        if zone.endswith("-FW") or zone.endswith("-KERNEL"):
            continue
        if args.zone_filter and args.zone_filter not in zone:
            continue
        by_rz[(risc, zone)].append((core, sum(d), len(d)))
    for risc in RISC_ORDER:
        items = sorted(
            ((z, v) for (rr, z), v in by_rz.items() if rr == risc),
            key=lambda kv: -statistics.median(x[1] for x in kv[1]),
        )
        for zone, v in items:
            tot = [x[1] for x in v]
            ex = statistics.median(x[2] for x in v)
            print(
                f"  {risc:8s} {zone:32s} {len(v):5d} {ex:9.0f} {statistics.median(tot):10.0f} {max(tot):10.0f} {min(tot):10.0f}"
            )

    # zone coverage vs kernel span (per RISC, worst core)
    print("\n== zone coverage: last user-zone end vs *-KERNEL end (cycles), per RISC (worst core) ==")
    for risc in RISC_ORDER:
        worst = None
        for (core, rr, zone), s in starts.items():
            if rr != risc or not zone.endswith("-KERNEL"):
                continue
            k_end = max(ends.get((core, rr, zone), [0]))
            k_start = min(s)
            user_end = max(
                (
                    max(ends[(c2, r2, z2)])
                    for (c2, r2, z2) in ends
                    if c2 == core and r2 == rr and not z2.endswith("-KERNEL") and not z2.endswith("-FW")
                ),
                default=k_start,
            )
            cov = (user_end - k_start) / (k_end - k_start) if k_end > k_start else 0
            if worst is None or cov < worst[0]:
                worst = (cov, core)
        if worst:
            print(f"  {risc:8s} min coverage {worst[0]*100:5.1f}% (core {worst[1]})")


if __name__ == "__main__":
    main()
