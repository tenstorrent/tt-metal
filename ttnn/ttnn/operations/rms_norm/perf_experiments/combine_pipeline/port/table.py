"""table.py <tag>: zip runs/<tag>_labels.jsonl with runs/<tag>_ops.csv GenericOp rows (dispatch order)."""
import csv, json, sys
from pathlib import Path

H = Path(__file__).parent / "runs"
tag = sys.argv[1]
recs = [json.loads(l) for l in open(H / f"{tag}_labels.jsonl")]
recs = [r for r in recs if r["pcc"] != 0.0 or r["max_abs_vs_torch"] != 0.0]  # drop the NO_DISPATCH collect pass
ops = [r for r in csv.DictReader(open(H / f"{tag}_ops.csv")) if r["OP CODE"].startswith("GenericOp")]
if len(ops) != len(recs):
    print(f"WARNING: {len(ops)} GenericOp rows vs {len(recs)} labels")
# labels are written per cell after all its variants ran, in dispatch order -> 1:1 with ops
base = {}
for r, o in zip(recs, ops):
    r["ns"] = int(o["DEVICE KERNEL DURATION [ns]"])
    if r["variant"] == "base":
        base[r["cell"]] = r["ns"]
print(
    "%-24s %-10s %4s %4s %3s %3s %4s %9s %7s %9s %9s %9s"
    % ("cell", "variant", "k", "b", "nb", "w", "crt", "ns", "vsbase", "pcc", "maxabs_t", "maxabs_b")
)
for r in recs:
    if "ns" not in r:
        continue
    b = base.get(r["cell"])
    print(
        "%-24s %-10s %4s %4s %3s %3s %4s %9d %7s %9.6f %9.4g %9.4g"
        % (
            r["cell"],
            r["variant"],
            r.get("k", ""),
            r.get("b", ""),
            r.get("nb", ""),
            r.get("w", ""),
            r.get("crt", ""),
            r["ns"],
            ("%.3f" % (r["ns"] / b)) if b else "",
            r["pcc"],
            r["max_abs_vs_torch"],
            r.get("max_abs_vs_base", -1),
        )
    )
