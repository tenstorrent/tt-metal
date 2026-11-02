"""table.py <tag> [profiled]: zip runs/<tag>_labels.jsonl with runs/<tag>_ops.csv GenericOp rows (dispatch order)."""
import csv, json, sys
from pathlib import Path

H = Path(__file__).parent / "runs"
tag = sys.argv[1]
prof = len(sys.argv) < 3 or sys.argv[2] == "1"
recs = [json.loads(l) for l in open(H / f"{tag}_labels.jsonl")]
recs = [r for r in recs if not (r["pcc"] != r["pcc"] and r["max_abs_vs_torch"] == 0)]  # drop the harness dry pass
ops = []
if prof and (H / f"{tag}_ops.csv").exists():
    ops = [
        r for r in csv.DictReader(open(H / f"{tag}_ops.csv")) if r.get("OP CODE", "GenericOp").startswith("GenericOp")
    ]
    if len(ops) != len(recs):
        print(f"WARNING: {len(ops)} GenericOp rows vs {len(recs)} labels")
base = {}
print(
    "%-26s %-8s %4s %4s %4s %10s %7s %9s %10s %9s %10s %s"
    % ("cell", "variant", "k", "U", "F", "ns", "vsbase", "pcc", "maxabs_t", "pcc_b", "maxabs_b", "ok")
)
for i, r in enumerate(recs):
    ns = int(ops[i]["DEVICE KERNEL DURATION [ns]"]) if i < len(ops) else 0
    if r["variant"] == "base":
        base[r["cell"]] = ns
    b = base.get(r["cell"])
    print(
        "%-26s %-8s %4s %4s %4s %10d %7s %9.6f %10.4g %9s %10s %s"
        % (
            r["cell"],
            r["variant"],
            r.get("k", ""),
            r.get("U", ""),
            r.get("F", "") if r.get("flat", True) else "nf",
            ns,
            ("%.3f" % (ns / b)) if (b and ns) else "",
            r["pcc"],
            r["max_abs_vs_torch"],
            ("%.6f" % r["pcc_vs_base"]) if "pcc_vs_base" in r else "",
            ("%.4g" % r["max_abs_vs_base"]) if "max_abs_vs_base" in r else "",
            r["ok"],
        )
    )
