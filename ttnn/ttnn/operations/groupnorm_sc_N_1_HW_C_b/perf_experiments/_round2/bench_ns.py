#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE KERNEL DURATION [ns] per op row of an ops_perf_results CSV, labelled with _op_bench.py case names
(the bench's collection order: test_bench, test_bench_sharded, test_bench_sharded_inplace, test_bench_guard,
or a -k subset passed as --names a,b,c).

usage: bench_ns.py <ops_perf_results.csv> [--names n1,n2,...] [--json out.json]
"""
import csv
import json
import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C_b")


def all_names():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_op_bench_names", "tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C_b/_op_bench.py"
    )
    src = open(spec.origin).read()
    ns = {}
    # cheap: exec only the case lists (they need ttnn layouts)
    import ttnn  # noqa

    for lst in ("CASES", "SHARDED_CASES", "SHARDED_INPLACE_CASES", "GUARD_CASES"):
        start = src.index(f"\n{lst} = [")
        end = src.index("\n]\n", start) + 3
        exec(src[start:end], {"ttnn": ttnn}, ns)
    return [c[0] for c in ns["CASES"] + ns["SHARDED_CASES"] + ns["SHARDED_INPLACE_CASES"] + ns["GUARD_CASES"]]


def main(argv):
    path = argv[0]
    names = None
    if "--names" in argv:
        names = argv[argv.index("--names") + 1].split(",")
    rows = [r for r in csv.DictReader(open(path)) if r.get("DEVICE KERNEL DURATION [ns]", "").strip()]
    if names is None:
        names = all_names()
    out = {}
    for i, r in enumerate(rows):
        nm = names[i] if i < len(names) else f"row{i}"
        out[nm] = int(float(r["DEVICE KERNEL DURATION [ns]"]))
    for k, v in out.items():
        print(f"{k:34s} {v:>12,d}")
    if "--json" in argv:
        json.dump(out, open(argv[argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
