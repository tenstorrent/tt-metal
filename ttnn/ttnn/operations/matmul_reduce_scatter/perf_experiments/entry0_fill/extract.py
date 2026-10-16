"""Per-replay op latency (max over devices of DEVICE KERNEL DURATION [ns]) from an ops_perf_results CSV.

usage: python extract.py <ops_perf_results.csv> [replays=3] [devices=4]
Prints the last `replays` calls (each = `devices` consecutive GenericOp rows), per device and the max.
"""
import csv
import sys

path = sys.argv[1]
nrep = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 4
rows = [r for r in csv.DictReader(open(path)) if r["OP CODE"].startswith("GenericOp")]
rows = rows[-nrep * ndev :]
maxes = []
for i in range(nrep):
    grp = rows[i * ndev : (i + 1) * ndev]
    per = {int(r["DEVICE ID"]): int(r["DEVICE KERNEL DURATION [ns]"]) for r in grp}
    maxes.append(max(per.values()))
    print(f"replay {i}: max {max(per.values())} ns  per-dev {dict(sorted(per.items()))}")
s = sorted(maxes)
print(f"OP_LATENCY_NS median {s[len(s) // 2]} min {s[0]} max {s[-1]}")
