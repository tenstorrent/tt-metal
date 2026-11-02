import csv, sys, re

rows = list(csv.DictReader(open(sys.argv[1])))
if rows and "OP CODE" in rows[0]:
    rows = [r for r in rows if r["OP CODE"].startswith("GenericOp")]
log = open(sys.argv[2]).read()
order = re.findall(r"HRF_ORDER (\S+) (\S+)", log)
order = order[len(order) - len(rows) :]  # the profiled (real) pass is the last one
print(f"generic rows {len(rows)} order {len(order)}")
res = {}
for (c, v), r in zip(order, rows):
    res.setdefault(c, {})[v] = int(r["DEVICE KERNEL DURATION [ns]"])
for c, d in res.items():
    b = d.get("base")
    s = "  ".join(f"{v}={n:>8d}" + (f" ({b / n:4.2f}x)" if b and v != "base" else "") for v, n in d.items())
    print(f"{c:28s} {s}")
