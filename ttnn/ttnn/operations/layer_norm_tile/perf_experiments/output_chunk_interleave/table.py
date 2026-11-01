"""Tabulate a bench.sh result dir: latency (reps=1) and per-row compute ((reps33 - reps1) / 32) per variant x shape."""

import sys
from pathlib import Path

out = Path(sys.argv[1])
ids = (out / "ids.txt").read_text().split()
ns = [int(x) for x in (out / "ns.txt").read_text().split()]
data = {}
for ident, val in zip(ids, ns):
    kv = dict(p.split("=") for p in ident.split(","))
    shape = f"R{kv['rows']}_Wt{kv['Wt']}_{'fp32' if kv['fp32_dest'] == 'True' else 'bf16'}"
    data.setdefault((kv["variant"], shape), {})[int(kv["reps"])] = val
variants = list(dict.fromkeys(v for v, _ in data))
shapes = list(dict.fromkeys(s for _, s in data))
print("latency reps=1 ns | per-row compute ns ((r33-r1)/32)")
print(f"{'variant':16s}" + "".join(f"{s:>24s}" for s in shapes))
for v in variants:
    cells = []
    for s in shapes:
        d = data.get((v, s), {})
        rows = int(s.split("_")[0][1:])
        per = (d[33] - d[1]) / 32 / rows if 33 in d and 1 in d else float("nan")
        cells.append(f"{d.get(1, 0):>12d} {per:>10.0f}")
    print(f"{v:16s}" + "".join(f"{c:>24s}" for c in cells))
