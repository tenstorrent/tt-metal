# usage: bench_report.py <ops_perf_results.csv> <tag>: per-variant timed ns (2nd GenericOp row of each variant) joined
# with the last run's records of results_bench.jsonl carrying <tag>; rewrites their ns / ns_per_tap in place.
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["OP CODE"].startswith("GenericOp")]
tag = sys.argv[2]
res = HERE / "results_bench.jsonl"
recs = [json.loads(l) for l in open(res)]
mine = [i for i, r in enumerate(recs) if r.get("tag") == tag and not r.get("ns")]
n = len(rows) // 2
mine = mine[-n:]
empty = None
for k, i in enumerate(mine):
    ns = float(rows[2 * k + 1]["DEVICE KERNEL DURATION [ns]"])
    ns0 = float(rows[2 * k]["DEVICE KERNEL DURATION [ns]"])
    r = recs[i]
    r["ns"], r["ns_first"] = ns, ns0
    r["ns_per_tap"] = ns / (r["taps"] * r["reps"])
    print(
        f"{tag:10s} {r['variant']:<10} {ns:>9.0f} ns ({ns0:>9.0f}) {r['ns_per_tap']:6.1f} ns/tap  exact={r['exact']}  "
        f"ct={r['ct']} W={r['W']} rows={r['rows']} reps={r['reps']} {r.get('dtype')}"
    )
with open(res, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
