# Pair the newest ops_perf_results CSV's last N GenericOp rows with results/perf_order.txt (last N lines).
import csv
import glob
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
csvs = sorted(
    glob.glob("/localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime
)
rows = list(csv.reader(open(csvs[-1])))
col = rows[0].index("DEVICE KERNEL DURATION [ns]")
durs = [int(float(r[col])) for r in rows[1:] if r and r[col]]
order = [l.split() for l in open(os.path.join(here, "results", "perf_order.txt")) if l.strip()]
n = int(sys.argv[1]) if len(sys.argv) > 1 else len(order)
order, durs = order[-n:], durs[-n:]
none = {}
for o, d in zip(order, durs):
    reps = int(o[1].split("=")[1])
    tiles = int(o[2].split("=")[1])
    print(f"{o[0]:8s} {d:9d} ns  {d / (reps * tiles):7.1f} ns/tile  {o[1]} {o[2]}  csv={os.path.basename(csvs[-1])}")
