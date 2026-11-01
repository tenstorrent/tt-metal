"""table.py <ops.csv> <log>: GenericOp DEVICE KERNEL DURATION rows labelled by the bench's `PB <cell> <variant>` lines."""
import csv, re, sys

g = [
    int(r["DEVICE KERNEL DURATION [ns]"])
    for r in csv.DictReader(open(sys.argv[1]))
    if r["OP CODE"].startswith("GenericOp")
]
lab = [(m.group(1), m.group(2)) for m in re.finditer(r"^PB (\S+) (\S+) pcc", open(sys.argv[2]).read(), re.M)]
lab = lab[-len(g) :]  # the up-front compile pass prints the same labels first
if len(g) != len(lab):
    print(f"WARNING: {len(g)} GenericOp rows vs {len(lab)} labels")
base = {}
for (c, v), ns in zip(lab, g):
    if v == "base":
        base.setdefault(c, ns)
    b = base.get(c, ns)
    print(f"{c:24s}{v:14s}{ns:>9d} ns  {b / ns:6.3f}x vs first base  ({100 * (ns - b) / b:+.1f}%)")
