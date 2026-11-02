"""Usage: parse_perf.py name1,name2,...  -> maps the last N GenericOp rows of the newest ops_perf_results CSV
to the names (pytest parametrize order) and prints DEVICE KERNEL DURATION [ns]. Optional env GST_DIV=<n> also
prints ns / n (per-tile cost)."""
import csv
import glob
import os
import sys

names = sys.argv[1].split(",")
path = max(
    glob.glob("/localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime
)
with open(path) as f:
    rows = [r for r in csv.DictReader(f) if r["OP CODE"].startswith("GenericOp")]
rows = rows[-len(names) :]
div = float(os.environ.get("GST_DIV", "0"))
print(f"# {path}")
for n, r in zip(names, rows):
    ns = float(r["DEVICE KERNEL DURATION [ns]"])
    extra = f"  per_unit={ns / div:.1f}" if div else ""
    print(f"GST_RESULT {n:12s} {ns:12.0f} ns  cores={r['CORE COUNT']}{extra}")
