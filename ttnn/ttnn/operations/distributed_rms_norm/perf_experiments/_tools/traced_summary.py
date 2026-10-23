"""Summarize a test_perf_ring --profile CSV: per traced call (the 6 replayed calls), the slowest device's DEVICE KERNEL
DURATION; prints per-shape median + range. usage: traced_summary.py <ops_perf_results csv> [n_calls=6]"""
import csv, sys, statistics, collections

path = sys.argv[1]
n_calls = int(sys.argv[2]) if len(sys.argv) > 2 else 6
rows = [r for r in csv.DictReader(open(path)) if "GenericOp" in r["OP CODE"] and r["DEVICE KERNEL DURATION [ns]"]]
by_dev = collections.defaultdict(list)
for r in rows:
    by_dev[r["DEVICE ID"]].append(int(r["DEVICE KERNEL DURATION [ns]"]))
# rows per device in call order: [warm-up + per shape: 1 untraced? ] -> take blocks of (1 + n_calls) per shape
devs = sorted(by_dev)
n = len(by_dev[devs[0]])
per = 1 + n_calls
for s in range(n // per):
    calls = [max(by_dev[d][s * per + 1 + i] for d in devs) for i in range(n_calls)]
    print(
        f"shape#{s}: slowest-device per traced call {calls} -> median {statistics.median(calls)/1000:.2f} us "
        f"(range {min(calls)/1000:.2f}-{max(calls)/1000:.2f})"
    )
    print(
        f"shape#{s}: slowest-device per traced call {calls} -> median {statistics.median(calls)/1000:.2f} us "
        f"(range {min(calls)/1000:.2f}-{max(calls)/1000:.2f})"
    )
