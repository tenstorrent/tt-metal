"""Join order.log (one line per bench test, in run order) with this run's ops_perf CSV (one generic op per test)."""
import csv
import glob
import os
import sys

marker, order_log = sys.argv[1], sys.argv[2]
since = os.path.getmtime(marker)
csvs = [
    p
    for p in glob.glob("/localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv")
    if os.path.getmtime(p) >= since
]
lines = [l.strip() for l in open(order_log)] if os.path.exists(order_log) else []
for path in sorted(csvs, key=os.path.getmtime, reverse=True):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("DEVICE KERNEL DURATION [ns]")]
    if 0 < len(rows) <= len(lines):
        lines = lines[-len(rows) :]
        print(f"# {path}")
        for line, r in zip(lines, rows):
            print(f"{line}|ns={r['DEVICE KERNEL DURATION [ns]']}|cores={r['CORE COUNT']}")
        break
else:
    print(f"# no matching CSV ({len(lines)} tests): {csvs}")
