# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-RISC zone TIMELINE of zoned runs (CP_ZONES=1): for each (RISC, zone, k-th instance on the core) the mean and
max start / end over Tensix cores, in ns from the op's earliest kernel start (all cores, all RISCs). Reads the
newest ops CSV + profile_log_device.csv; labels from bench_order.txt (last N lines).

    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/combine_pipeline/cp_timeline.py [--risc TRISC_0,NCRISC]
"""

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
FW = ("BRISC-FW", "NCRISC-FW", "TRISC-FW", "ERISC-FW")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--log", default=str(HERE / "bench_order.txt"))
    ap.add_argument("--device-log", default="generated/profiler/.logs/profile_log_device.csv")
    ap.add_argument("--risc", default="TRISC_0,NCRISC,BRISC")
    a = ap.parse_args()
    path = a.csv or max(glob.glob("generated/profiler/reports/*/ops_perf_results*.csv"), key=os.path.getmtime)
    rows = [r for r in csv.DictReader(open(path)) if "GenericOp" in r["OP CODE"]]
    labels = [l.strip() for l in open(a.log) if l.strip()][-len(rows) :]
    run_label = {r["GLOBAL CALL COUNT"]: (lab, r["DEVICE KERNEL DURATION [ns]"]) for lab, r in zip(labels, rows)}
    with open(a.device_log) as fh:
        header = fh.readline()
        mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        drows = [r for r in csv.DictReader(fh, skipinitialspace=True) if r["run host ID"] in run_label]
    by_run = defaultdict(list)
    for r in drows:
        by_run[r["run host ID"]].append(r)
    riscs = a.risc.split(",")
    for run, rr in by_run.items():
        lab, dur = run_label[run]
        t0 = min(int(r["time[cycles since reset]"]) for r in rr)
        stack = defaultdict(list)
        inst = defaultdict(int)
        spans = defaultdict(list)  # (risc, zone, k) -> [(start, end)]
        for r in rr:
            risc = r["RISC processor type"]
            name = r["zone name"]
            if name in FW or risc not in riscs:
                continue
            key = (r["core_x"], r["core_y"], risc)
            cyc = int(r["time[cycles since reset]"]) - t0
            if r["type"] == "ZONE_START":
                stack[(key, name)].append(cyc)
            elif r["type"] == "ZONE_END" and stack[(key, name)]:
                s = stack[(key, name)].pop()
                k = inst[(key, name)]
                inst[(key, name)] += 1
                spans[(risc, name, k)].append((s, cyc))
        print(f"\n## {lab}  ({dur} ns)   [ns from first kernel start: start mean/max -> end mean/max, dur mean]")
        f = 1000.0 / mhz
        for risc in riscs:
            items = sorted(
                ((k, v) for k, v in spans.items() if k[0] == risc), key=lambda kv: sum(s for s, _ in kv[1]) / len(kv[1])
            )
            for (rs, name, k), v in items:
                ss = [s * f for s, _ in v]
                ee = [e * f for _, e in v]
                dd = [(e - s) * f for s, e in v]
                print(
                    f"  {rs:<8} {name:<16} #{k:<2} n={len(v):<3} {sum(ss) / len(ss):8.0f} {max(ss):8.0f} -> "
                    f"{sum(ee) / len(ee):8.0f} {max(ee):8.0f}  dur {sum(dd) / len(dd):7.0f}"
                )


if __name__ == "__main__":
    main()
