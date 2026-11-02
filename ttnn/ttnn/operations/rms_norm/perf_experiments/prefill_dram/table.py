"""Per-cell base vs variant DEVICE KERNEL DURATION from a bench ops CSV (PD_CELLS / PD_VARIANTS from env)."""
import csv, os, sys

cells = os.environ["PD_CELLS"].split(",")
vs = ["base"] + os.environ.get("PD_VARIANTS", "clone").split(",")
g = [r for r in csv.DictReader(open(sys.argv[1])) if r["OP CODE"].startswith("GenericOp")]
i = 0
for c in cells:
    b = None
    for v in vs:
        if i >= len(g):
            break
        ns = int(g[i]["DEVICE KERNEL DURATION [ns]"])
        i += 1
        b = b or ns
        print(f"{c:22s} {v:12s} {ns:>9d} ns  {b/ns:5.3f}x")
