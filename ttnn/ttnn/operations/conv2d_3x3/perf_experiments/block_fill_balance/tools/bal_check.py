"""Offline: op_copy's _balance_fill_nocs on one column of a case vs noc_lp's LP optimum (same resource model)."""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import linkload  # noqa: E402
import noc_lp  # noqa: E402
import pdstub  # noqa: E402

pdc = pdstub.load(str(HERE.parent / "op_copy" / "conv2d_3x3_program_descriptor.py"), "pdc")
case = sys.argv[1] if len(sys.argv) > 1 else "aspp1"
pcs, row_bytes, nrows = noc_lp.pieces(case)
column = []
for m in range(nrows):
    column.append((2 + m, [(2 + s, c) for mm, s, c in pcs if mm == m]))
t = time.time()
x1 = pdc._balance_fill_nocs(column, 12, row_bytes)
dt = time.time() - t
flat = [v for xs in x1 for v in xs]
res = noc_lp.report("greedy", pcs, flat, row_bytes)
noc_lp.report("lp", pcs, noc_lp.lp(pcs, row_bytes), row_bytes)
print(
    f"  greedy {dt * 1e3:.0f} ms; NoC1 rows per core:",
    [sum(xs) for xs in x1],
    "of",
    [sum(c for _, c in l) for _, l in column],
)
for m, xs in enumerate(x1):
    print(f"   m{m}: " + " ".join(f"s{s}:{x}/{c}" for (mm, s, c), x in zip([p for p in pcs if p[0] == m], xs)))
