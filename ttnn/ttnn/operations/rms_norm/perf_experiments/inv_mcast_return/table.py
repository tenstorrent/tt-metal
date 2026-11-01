"""usage: table.py labels.txt ops.csv -> per cell: device kernel ns per variant (+ delta vs base)."""
import csv, sys, collections

labels = [l.strip() for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in csv.DictReader(open(sys.argv[2])) if r["OP CODE"].startswith("GenericOp")]
assert len(rows) == len(labels), (len(rows), len(labels))
t = collections.OrderedDict()
for lab, r in zip(labels, rows):
    cell, v = lab.split(":")
    t.setdefault(cell, {})[v] = int(r["DEVICE KERNEL DURATION [ns]"])
vs = []
for d in t.values():
    for v in d:
        if v not in vs:
            vs.append(v)
print("%-22s" % "cell" + "".join("%12s" % v for v in vs) + "   best-vs-base")
for c, d in t.items():
    b = d.get("base")
    best = min((d[v], v) for v in d if v not in ("base", "uc"))
    print("%-22s" % c + "".join("%12s" % d.get(v, "-") for v in vs) + "   %s %.3fx" % (best[1], b / best[0]))
