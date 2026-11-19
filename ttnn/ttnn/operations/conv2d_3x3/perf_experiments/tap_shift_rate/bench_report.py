# usage: bench_report.py <ops_perf_results.csv> <tag>: per-variant timed ns (2nd GenericOp row of each variant) joined
# with the last run's records of results_bench.jsonl carrying <tag>; rewrites their ns / ns_per_tap in place.
import csv, json, sys
from pathlib import Path

HERE = Path(__file__).parent
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["OP CODE"].startswith("GenericOp")]
tag = sys.argv[2]
res = HERE / "results_bench.jsonl"
recs = [json.loads(l) for l in open(res)]
mine = [i for i, r in enumerate(recs) if r.get("tag") == tag and not r.get("ns")]
n = len(rows) // 2
mine = mine[-n:]
base = None
for k, i in enumerate(mine):
    ns = float(rows[2 * k + 1]["DEVICE KERNEL DURATION [ns]"])
    ns0 = float(rows[2 * k]["DEVICE KERNEL DURATION [ns]"])
    r = recs[i]
    r["ns"], r["ns_first"] = ns, ns0
    r["ns_per_tap"] = ns / (r["taps"] * r["reps"])
    if r["variant"] == "base":
        base = ns
    sp = f"{base / ns:5.3f}x" if base else "  -  "
    print(
        f"{tag:10s} {r['variant']:<14} {ns:>9.0f} ns ({ns0:>9.0f}) {sp} {r['ns_per_tap']:6.1f} ns/tap  exact={r['exact']}  d={r['d']} ct={r['ct']}"
    )
with open(res, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
