"""Summarize a test_perf_ring_sharded --profile CSV: per case 6 untraced warm-up calls + 6 traced calls; slowest device
per traced call, median + range. usage: sharded_summary.py <ops_perf_results csv>"""
import csv, sys, statistics, collections

rows = [
    r for r in csv.DictReader(open(sys.argv[1])) if "GenericOp" in r["OP CODE"] and r["DEVICE KERNEL DURATION [ns]"]
]
by_dev = collections.defaultdict(list)
for r in rows:
    by_dev[r["DEVICE ID"]].append(int(r["DEVICE KERNEL DURATION [ns]"]))
devs = sorted(by_dev)
n = min(len(by_dev[d]) for d in devs)
for s in range(n // 12):
    calls = [max(by_dev[d][s * 12 + 6 + i] for d in devs) for i in range(6)]
    print(
        f"case#{s}: median {statistics.median(calls)/1000:.2f} us (range {min(calls)/1000:.2f}-{max(calls)/1000:.2f})"
    )
