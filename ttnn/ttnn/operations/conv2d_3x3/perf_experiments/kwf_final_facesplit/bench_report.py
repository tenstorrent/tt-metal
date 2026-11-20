# usage: bench_report.py <ops_perf_results.csv> <tag>: per-variant timed ns (2nd GenericOp row of each variant) joined
# with the records of results_bench.jsonl carrying <tag>; writes ns / cycles per tile (1350 MHz) in place.
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
for k, i in enumerate(mine):
    ns = float(rows[2 * k + 1]["DEVICE KERNEL DURATION [ns]"])
    r = recs[i]
    r["ns"] = ns
    r["cyc_per_tile"] = ns * 1.35 / r["tiles"]
    print(f"{tag:10s} {r['variant']:<14} {ns:>9.0f} ns  {r['cyc_per_tile']:6.1f} cycles/tile  tiles={r['tiles']}")
with open(res, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
