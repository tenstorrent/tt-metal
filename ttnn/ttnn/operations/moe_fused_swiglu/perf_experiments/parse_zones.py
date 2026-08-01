# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read the newest `--profile` report and print (a) whole-op DEVICE KERNEL DURATION per call and
(b) per-`MaybeDeviceZoneScope` stage statistics across the grid.

    python3 ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/parse_zones.py [report_dir]

The device log is a flat list of BEGIN/END timestamp records:
    PCIe slot, core_x, core_y, RISC type, timer_id, time[cycles], data, run host ID, trace id,
    trace id counter, zone name, type, source line, source file, meta data
A stage's duration is END - BEGIN on the same (core, risc, zone), matched in arrival order, which is
correct because a zone never nests inside itself here (one `{ }` block per stage).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPORTS = Path("generated/profiler/reports")


def newest_report(explicit=None):
    if explicit:
        return Path(explicit)
    dirs = sorted((d for d in REPORTS.iterdir() if d.is_dir()), key=lambda d: d.name)
    for d in reversed(dirs):
        if (d / "profile_log_device.csv").exists():
            return d
    raise SystemExit(f"no report with profile_log_device.csv under {REPORTS}")


def whole_op(report):
    rows = []
    for csv_path in report.glob("ops_perf_results*.csv"):
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                if r.get("OP CODE") != "GenericOpDeviceOperation":
                    continue
                rows.append(
                    (
                        int(r["GLOBAL CALL COUNT"]),
                        int(r["DEVICE KERNEL DURATION [ns]"]),
                        int(r["CORE COUNT"]),
                    )
                )
    return sorted(rows)


def zones(report):
    freq_mhz = 1350.0
    path = report / "profile_log_device.csv"
    with open(path) as fh:
        head = fh.readline()
        if "CHIP_FREQ[MHz]" in head:
            try:
                freq_mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
            except Exception:
                pass
        reader = csv.reader(fh)
        cols = [c.strip() for c in next(reader)]
        idx = {c: i for i, c in enumerate(cols)}
        open_ts = defaultdict(list)
        durs = defaultdict(list)
        for row in reader:
            if len(row) < len(cols):
                continue
            zone = row[idx["zone name"]].strip()
            kind = row[idx["type"]].strip()
            if not zone:
                continue
            key = (
                row[idx["core_x"]].strip(),
                row[idx["core_y"]].strip(),
                row[idx["RISC processor type"]].strip(),
                zone,
            )
            ts = int(row[idx["time[cycles since reset]"]])
            if kind == "ZONE_START":
                open_ts[key].append(ts)
            elif kind == "ZONE_END" and open_ts[key]:
                durs[key].append(ts - open_ts[key].pop())
        per_zone = defaultdict(list)  # zone -> per-core TOTAL ns
        per_zone_calls = defaultdict(int)
        per_core = defaultdict(dict)  # (risc, cx, cy) -> {zone: ns}
        for (cx, cy, risc, zone), ds in durs.items():
            ns = sum(ds) * 1000.0 / freq_mhz
            per_zone[(risc, zone)].append(ns)
            per_zone_calls[(risc, zone)] += len(ds)
            per_core[(risc, cx, cy)][zone] = ns
        return per_zone, per_zone_calls, per_core


def main():
    report = newest_report(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"report: {report}")
    print("\n-- whole op (DEVICE KERNEL DURATION) --")
    for call, ns, cores in whole_op(report):
        print(f"  call {call:>6}  {ns:>10,} ns   cores={cores}")
    per_zone, calls, per_core = zones(report)
    if not per_zone:
        print("\n(no device zone records)")
        return
    print("\n-- per-stage zones: per-core TOTAL ns for the whole dispatch --")
    print(f"  {'risc':<8}{'stage':<24}{'cores':>6}{'mean':>12}{'max':>12}{'min':>12}{'recs':>8}")
    for (risc, zone), vals in sorted(per_zone.items(), key=lambda kv: -max(kv[1])):
        print(
            f"  {risc:<8}{zone:<24}{len(vals):>6}{sum(vals) / len(vals):>12,.0f}"
            f"{max(vals):>12,.0f}{min(vals):>12,.0f}{calls[(risc, zone)]:>8}"
        )

    # PER-CORE compute decomposition. The reduce ROOTS are the cores whose `compute_reduce` is
    # large; every other core's long stage is a `cb_wait`, not work. Sorted by the SUM of the
    # kernel's real work stages so the critical cores come first.
    work = ["compute_tilize", "compute_gateup", "compute_reduce", "compute_swiglu", "compute_down", "compute_out_pack"]
    rows = []
    for (risc, cx, cy), zs in per_core.items():
        if risc != "TRISC_1":
            continue
        rows.append(((cx, cy), zs, zs.get("TRISC-KERNEL", 0.0)))
    if rows:
        print("\n-- TRISC_1 per-core work decomposition (top 14 by compute_reduce) --")
        hdr = f"  {'core':<10}{'KERNEL':>10}" + "".join(f"{w.replace('compute_', ''):>10}" for w in work)
        print(hdr)
        rows.sort(key=lambda r: -r[1].get("compute_reduce", 0.0))
        for (cx, cy), zs, total in rows[:14]:
            line = f"  {cx + ',' + cy:<10}{total:>10,.0f}"
            line += "".join(f"{zs.get(w, 0.0):>10,.0f}" for w in work)
            print(line)


if __name__ == "__main__":
    main()
