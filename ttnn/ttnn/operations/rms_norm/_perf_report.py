# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf report for rms_norm bench runs (test_rms_norm_perf_bench.py under run_safe_pytest.sh --profile).

  * per-test DEVICE KERNEL DURATION [ns] from the newest ops_perf_results CSV, labelled with RMS_BENCH_LOG;
  * for every `zones` test: per-(zone, RISC) per-core totals off profile_log_device.csv — max over Tensix cores
    (the critical-path reading) and mean — plus the two marker-budget integrity checks (zone exhaustion is
    SILENT, kernel_lib/perf_instrumentation.hpp): markers per (core, RISC) vs the 250 cap, and the last user
    zone's end vs that RISC's KERNEL span.

Usage:
    python3 -m ttnn.operations.rms_norm._perf_report [--csv <ops_perf_results.csv>] [--log <order file>]
        [--device-log <profile_log_device.csv>] [--no-zones]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

MARKER_CAP = 250
FW = ("BRISC-FW", "NCRISC-FW", "TRISC-FW", "ERISC-FW")


def newest_ops_csv():
    files = glob.glob("generated/profiler/reports/*/ops_perf_results*.csv")
    return max(files, key=os.path.getmtime)


def durations(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    rows = [r for r in rows if "GenericOp" in r["OP CODE"]]
    return rows


def zone_tables(dev_log, wanted_runs):
    """run host ID -> {(zone, risc): [per-core totals ns]}, coverage notes."""
    with open(dev_log) as fh:
        header = fh.readline()
        mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader(fh, skipinitialspace=True))
    by_run = defaultdict(list)
    for r in rows:
        if r["run host ID"] in wanted_runs:
            by_run[r["run host ID"]].append(r)
    out = {}
    for run, rr in by_run.items():
        markers = defaultdict(int)
        stack = defaultdict(list)
        per_core = defaultdict(lambda: defaultdict(float))
        cnt = defaultdict(int)
        kspan = {}
        last_end = defaultdict(int)
        kend = {}
        for r in rr:
            key = (r["core_x"], r["core_y"], r["RISC processor type"])
            markers[key] += 1
            name = r["zone name"]
            cyc = int(r["time[cycles since reset]"])
            if r["type"] == "ZONE_START":
                stack[(key, name)].append(cyc)
            elif r["type"] == "ZONE_END":
                st = stack[(key, name)]
                if not st:
                    continue
                start = st.pop()
                dur = (cyc - start) / mhz * 1000.0
                if name in FW:
                    continue
                if name.endswith("-KERNEL"):
                    kspan[key] = dur
                    kend[key] = cyc
                    continue
                per_core[(name, key[2])][key[:2]] += dur
                cnt[(name, key[2])] += 1
                last_end[key] = max(last_end[key], cyc)
        capped = [k for k, v in markers.items() if v >= MARKER_CAP - 4]
        short = []
        for k, e in kend.items():
            if k in last_end and kspan[k] > 0:
                gap = (e - last_end[k]) / mhz * 1000.0
                if gap > 0.25 * kspan[k] and gap > 2000:
                    short.append((k, gap, kspan[k]))
        spans = defaultdict(list)
        for k, v in kspan.items():
            spans[k[2]].append(v)
        out[run] = (per_core, cnt, capped, short, spans)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--log", default="generated/rms_norm_bench_order.txt")
    ap.add_argument("--device-log", default="generated/profiler/.logs/profile_log_device.csv")
    ap.add_argument("--no-zones", action="store_true")
    a = ap.parse_args()
    path = a.csv or newest_ops_csv()
    rows = durations(path)
    labels = [l.strip() for l in open(a.log) if l.strip()] if os.path.exists(a.log) else []
    print(f"# {path}: {len(rows)} GenericOp rows, {len(labels)} labels")
    # run_safe_pytest.sh's precompile pass runs the test bodies first (and logs labels): the profiled pass is the
    # LAST len(rows) labels.
    labels = labels[-len(rows) :] if rows else labels
    zone_runs = {}
    for i, r in enumerate(rows):
        lab = labels[i] if i < len(labels) else f"#{i}"
        print(
            f"{lab:<48} {float(r['DEVICE KERNEL DURATION [ns]']):>12.0f} ns  cores {r['CORE COUNT']:>4}"
            f"  BR {r['DEVICE BRISC KERNEL DURATION [ns]']:>9} NC {r['DEVICE NCRISC KERNEL DURATION [ns]']:>9}"
            f" T0 {r['DEVICE TRISC0 KERNEL DURATION [ns]']:>9} T1 {r['DEVICE TRISC1 KERNEL DURATION [ns]']:>9}"
            f" T2 {r['DEVICE TRISC2 KERNEL DURATION [ns]']:>9}"
        )
        if lab.endswith("|zones"):
            zone_runs[r["GLOBAL CALL COUNT"]] = lab
    if a.no_zones or not zone_runs or not os.path.exists(a.device_log):
        return
    tables = zone_tables(a.device_log, set(zone_runs))
    for run, lab in zone_runs.items():
        if run not in tables:
            print(f"\n## {lab}: run host ID {run} not in device log")
            continue
        per_core, cnt, capped, short, spans = tables[run]
        print(f"\n## zones {lab} (run host ID {run})")
        print("  kernel span max per RISC: " + ", ".join(f"{k} {max(v):.0f}" for k, v in sorted(spans.items())))
        print(f"  {'zone':<18} {'risc':<7} {'cores':>5} {'execs':>6} {'max/core ns':>12} {'mean/core ns':>13}")
        for (name, risc), d in sorted(per_core.items(), key=lambda kv: (kv[0][1], -max(kv[1].values()))):
            vals = list(d.values())
            print(
                f"  {name:<18} {risc:<7} {len(vals):>5} {cnt[(name, risc)]:>6} {max(vals):>12.0f}"
                f" {sum(vals) / len(vals):>13.0f}"
            )
        print(f"  marker-cap check: {len(capped)} (core,RISC) at the cap; short coverage: {len(short)}")
        for k, gap, sp in short[:5]:
            print(f"    core{k[:2]} {k[2]}: last zone ends {gap:.0f} ns before kernel end (span {sp:.0f})")


if __name__ == "__main__":
    main()
