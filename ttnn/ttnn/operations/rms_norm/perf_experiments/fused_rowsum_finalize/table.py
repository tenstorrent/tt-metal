"""Map GenericOp rows of an ops CSV to the (cell, variant) order logged by the bench; print a ns table.
usage: table.py ops.csv order.jsonl   (uses the LAST len(GenericOp rows) entries of order.jsonl: the up-front
collect pass logs the order too but dispatches nothing to the profiler)"""
import csv, json, sys, collections

rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["OP CODE"].startswith("GenericOp")]
order = [json.loads(l) for l in open(sys.argv[2])][-len(rows) :]
tab = collections.OrderedDict()
variants = []
for r, o in zip(rows, order):
    d = tab.setdefault(o["cell"], {})
    v = o["variant"]
    k = 1
    while (v if k == 1 else "%s#%d" % (v, k)) in d:
        k += 1
    v = v if k == 1 else "%s#%d" % (v, k)
    d[v] = int(r["DEVICE KERNEL DURATION [ns]"])
    if v not in variants:
        variants.append(v)
print("%-28s" % "cell" + "".join("%11s" % v for v in variants) + "   best/op")
for cell, d in tab.items():
    base = d.get("op") or d.get("0")
    best = min(d.values())
    print("%-28s" % cell + "".join("%11s" % d.get(v, "-") for v in variants) + "   %.3fx" % (base / best))
