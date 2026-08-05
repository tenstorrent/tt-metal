# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join perf_manifest.csv (launch order) against DEVICE KERNEL DURATION [ns].

    python3 <this dir>/read_perf.py [ops_perf_results_*.csv]

Every manifest row is exactly ONE `ttnn.generic_op` launch, in order; the profiler CSV
is filtered to the generic-op rows so host-side tensor traffic cannot shift the join.
"""
import csv
import glob
import pathlib
import sys

HERE = pathlib.Path(__file__).parent


def main():
    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"))[-1]
    )
    man = [r for r in csv.reader(open(HERE / "perf_manifest.csv")) if r]
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    ns = [
        int(r["DEVICE KERNEL DURATION [ns]"])
        for r in rows
        if r.get("DEVICE KERNEL DURATION [ns]", "").strip().isdigit()
    ]
    # `run_safe_pytest.sh --profile` PRECOMPILES by running the same selection once
    # unprofiled first, so the manifest holds an integer number of identical passes;
    # the profiled launches are the LAST one.
    if len(man) > len(ns) and len(ns) and len(man) % len(ns) == 0:
        man = man[-len(ns) :]
    if len(ns) != len(man):
        print(f"WARNING: {len(ns)} profiled launches vs {len(man)} manifest rows")
        print("  op codes:", {r.get("OP CODE") for r in rows})
    n = min(len(ns), len(man))

    rec = {}
    order = []
    for (case, option), t in zip(man[:n], ns[:n]):
        if case not in rec:
            order.append(case)
            rec[case] = {}
        rec[case][option] = t

    for case in order:
        e = rec[case]
        base = e.get("baseline")
        print(f"\n{case}   baseline = {base} ns")
        for opt, t in e.items():
            if opt == "baseline":
                continue
            sp = base / t if (base and t) else float("nan")
            print(f"    {opt:18s} {t:9d} ns   {sp:5.3f}x   ({t - base:+d} ns)")


if __name__ == "__main__":
    main()
