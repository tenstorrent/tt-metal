"""table.py <tag>: zip runs/<tag>_labels.jsonl (real-pass rows) with runs/<tag>_ops.csv GenericOp rows."""
import csv, json, sys
from pathlib import Path

H = Path(__file__).parent / "runs"
tag = sys.argv[1]
recs = [json.loads(l) for l in open(H / f"{tag}_labels.jsonl")]
recs = [r for r in recs if r["pcc"] != 0.0 or r.get("max_abs_vs_torch", 0) != 0.0]
ops = [r for r in csv.DictReader(open(H / f"{tag}_ops.csv")) if r["OP CODE"].startswith("GenericOp")]
if len(ops) != len(recs):
    print(f"WARNING: {len(ops)} GenericOp rows vs {len(recs)} labels")
base = {}
for r, o in zip(recs, ops):
    r["ns"] = int(o["DEVICE KERNEL DURATION [ns]"])
    if r["variant"] == "base":
        base[r["cell"]] = r["ns"]
print(
    "%-22s %-14s %4s %4s %5s %9s %7s %9s %10s %10s"
    % ("cell", "variant", "k", "b", "pipe", "ns", "vsbase", "pcc", "maxabs_t", "maxabs_b")
)
for r in zip(recs, ops):
    r = r[0]
    b = base.get(r["cell"])
    print(
        "%-22s %-14s %4s %4s %5s %9d %7s %9.6f %10.4g %10.4g"
        % (
            r["cell"],
            r["variant"],
            r.get("k", ""),
            r.get("b", ""),
            r.get("pipe", ""),
            r["ns"],
            ("%.3f" % (r["ns"] / b)) if b else "",
            r["pcc"],
            r["max_abs_vs_torch"],
            r.get("max_abs_vs_base", -1),
        )
    )
