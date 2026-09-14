# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: CB readiness times per reader ordering, from generated/profiler/.logs/profile_log_device.csv.

For every run id in the log (one per generic_op call, in execution order) and every active core, takes
the NCRISC *-KERNEL start as t=0 and reports, as median / max across cores (ns):
    x0_ready      ZONE_START of mark_x0_ready      (x block 0 pushed)
    gamma_ready   ZONE_START of mark_gamma_ready
    scaler_ready  ZONE_START of mark_scaler_ready
    wr_x0_seen    ZONE_START of mark_wr_x0_seen on BRISC, relative to the BRISC kernel start (consumer view)
    ncrisc_span   NCRISC kernel span (reader done)
plus the scaler-fill duration (rd_scaler) so L1-contention inflation is visible.

    python3 avail_report.py [--csv PATH] [names...]
"""

import argparse
import os
import statistics
from collections import defaultdict

DEFAULT_NAMES = [
    "baseline",
    "xfirst_one_barrier",
    "xfirst_split_barriers",
    "xfirst_trid_barriers",
    "scaler_x_gamma",
    "scaler_xg_one_barrier",
    "scaler_xg_trid",
    "x_gamma_scaler_last",
    "x_scaler_gamma",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv"),
    )
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    names = args.names or DEFAULT_NAMES

    lines = open(args.csv).read().splitlines()
    freq = 1000.0
    for part in lines[0].split(","):
        if "CHIP_FREQ" in part:
            freq = float(part.split(":")[1])
    ns = 1000.0 / freq
    rows = [[x.strip() for x in ln.split(",")] for ln in lines[2:] if ln.strip()]
    rows = [r for r in rows if len(r) >= 12 and r[7].isdigit()]
    run_ids = sorted({int(r[7]) for r in rows})

    print(
        f"{'variant':24s} {'x0_ready':>16s} {'gamma_ready':>16s} {'scaler_ready':>16s} {'wr_x0_seen':>16s} {'rd_scaler':>12s} {'ncrisc_span':>14s}"
    )
    print(f"{'':24s} {'med/max':>16s} {'med/max':>16s} {'med/max':>16s} {'med/max':>16s} {'med':>12s} {'med/max':>14s}")
    for i, rid in enumerate(run_ids):
        rr = [r for r in rows if int(r[7]) == rid]
        k_start = {}  # (core, risc) -> kernel start cycles
        k_end = {}
        marks = defaultdict(dict)  # (core, risc) -> zone -> first ZONE_START cycles
        zdur = defaultdict(list)
        zs = {}
        for r in rr:
            core, risc, cyc, zone, typ = (int(r[1]), int(r[2])), r[3], int(r[5]), r[10], r[11]
            key = (core, risc)
            if zone.endswith("-KERNEL"):
                if typ == "ZONE_START":
                    k_start[key] = cyc
                elif typ == "ZONE_END":
                    k_end[key] = cyc
            elif zone.startswith("mark_") and typ == "ZONE_START":
                marks[key].setdefault(zone, cyc)
            elif zone == "rd_scaler":
                if typ == "ZONE_START":
                    zs[key] = cyc
                elif typ == "ZONE_END" and key in zs:
                    zdur[key].append((cyc - zs[key]) * ns)

        def rel(zone, risc):
            vals = []
            for (core, rsc), m in marks.items():
                if rsc == risc and zone in m and (core, rsc) in k_start:
                    vals.append((m[zone] - k_start[(core, rsc)]) * ns)
            return vals

        def fmt(vals):
            return f"{statistics.median(vals):7.0f}/{max(vals):7.0f}" if vals else f"{'-':>15s}"

        spans = [(k_end[k] - k_start[k]) * ns for k in k_start if k in k_end and k[1] == "NCRISC"]
        sc = [d[0] for d in zdur.values() if d]
        name = names[i] if i < len(names) else f"run{rid}"
        print(
            f"{name:24s} {fmt(rel('mark_x0_ready', 'NCRISC')):>16s} {fmt(rel('mark_gamma_ready', 'NCRISC')):>16s} "
            f"{fmt(rel('mark_scaler_ready', 'NCRISC')):>16s} {fmt(rel('mark_wr_x0_seen', 'BRISC')):>16s} "
            f"{(statistics.median(sc) if sc else 0):12.0f} {fmt(spans):>14s}"
        )


if __name__ == "__main__":
    main()
