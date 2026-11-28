# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Median DEVICE KERNEL DURATION [ns] of one bench arm: rows of the given op code in a profiler CSV,
launch 1 skipped. Usage: python3 bench_parse.py <ops_perf_results.csv> [op_code_substring]"""

import csv
import statistics
import sys

path = sys.argv[1]
code = sys.argv[2] if len(sys.argv) > 2 else "GenericOp"
rows = [r for r in csv.DictReader(open(path)) if code.lower() in (r.get("OP CODE") or "").lower()]
ns = [int(float(r["DEVICE KERNEL DURATION [ns]"])) for r in rows]
cores = sorted({r["CORE COUNT"] for r in rows})
if len(ns) < 2:
    print(f"BENCH_PARSE too few rows ({len(ns)}) for {code} in {path}")
    sys.exit(1)
timed = ns[1:]
print(
    f"BENCH_PARSE n={len(timed)} median={statistics.median(timed):.0f} min={min(timed)} max={max(timed)} "
    f"cores={','.join(cores)} first={ns[0]}"
)
