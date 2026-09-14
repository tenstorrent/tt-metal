# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: per-variant MATH-thread (TRISC_1) ns of the FIN_* zones from a --profile run.

    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/finalize_col0_sfpu/parse_zones.py [--reps 32] [--csv PATH]

Pairs ZONE_START/ZONE_END per (core, RISC, zone, run id); for each FIN_* zone takes the LAST run id present
(the CSV accumulates across dispatches) and prints ns per finalize call = zone_ns / reps, alongside the
vector-op count so ns/vector is visible. Never touches the device.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import DEST_MODES, INIT_MODES, INITS, VARIANTS, VECTORS, zone_name  # noqa: E402


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
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--risc", default="TRISC_1")
    args = ap.parse_args()

    rows, freq_mhz = read_rows(args.csv)
    ns_per_cycle = 1000.0 / freq_mhz
    starts, ends = defaultdict(list), defaultdict(list)
    for r in rows:
        risc, cyc, zone, typ = r[3], int(r[5]), r[10], r[11]
        if risc != args.risc or not zone.startswith("FIN_") or not r[7].isdigit():
            continue
        key = (zone, int(r[7]))
        (starts if typ == "ZONE_START" else ends if typ == "ZONE_END" else defaultdict(list))[key].append(cyc)

    # zone -> ns of the last run id present
    last = {}
    for (zone, run_id), s in starts.items():
        e = ends.get((zone, run_id))
        if not e:
            continue
        d = (min(e) - min(s)) * ns_per_cycle
        if zone not in last or run_id > last[zone][0]:
            last[zone] = (run_id, d)

    print(f"CHIP_FREQ {freq_mhz:.0f} MHz, risc={args.risc}, reps={args.reps}, zones found={len(last)}")
    print(
        f"{'variant':18s} {'init':6s} {'dest':7s} {'vec':>4s} {'inits':>5s} {'zone_ns':>10s} {'ns/call':>9s} {'ns/vec':>7s} {'x_base':>7s}"
    )
    for approx in (False, True):
        for dest in DEST_MODES:
            base = last.get(zone_name("baseline", "pc", dest, approx), (None, None))[1]
            for init in INIT_MODES:
                for v in VARIANTS:
                    z = zone_name(v, init, dest, approx)
                    if z not in last:
                        continue
                    d = last[z][1]
                    per_call = d / args.reps
                    speed = (base / d) if base else float("nan")
                    print(
                        f"{v:18s} {init:6s} {dest + ('/apx' if approx else ''):11s} {VECTORS[v]:4d} {INITS[v]:5d} "
                        f"{d:10.0f} {per_call:9.1f} {per_call / VECTORS[v]:7.1f} {speed:7.2f}"
                    )


if __name__ == "__main__":
    main()
