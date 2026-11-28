# small_tensor_split: per-cell op (s1) vs candidate vs stock from results.jsonl (median over every
# recorded run of the (cell, variant); a run = median of launches 2..12 of one process pass).
#   python3 summarize.py [results.jsonl] [--all]     --all: every variant of every cell
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

path = (
    Path(sys.argv[1])
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
    else Path(__file__).parent / "results.jsonl"
)
show_all = "--all" in sys.argv

runs = defaultdict(list)
meta = {}
for line in path.read_text().splitlines():
    r = json.loads(line)
    if r.get("ns") is None or r.get("tag") in ("smoke", "keys", "keys2", "explore1", "check2", "check3", "dbg", "dbg2"):
        continue
    if r["variant"] != "stock" and r.get("exact") is False:
        continue
    runs[(r["cell"], r["variant"])].append(r["ns"])
    if r.get("regime"):
        meta[r["cell"]] = (r.get("shape"), r["regime"])

# The recommended rule's variant per cell (the rule maps every other cell to s1, i.e. the op).
CANDIDATE = {}
for cell, variant in runs:
    if variant == "rec":
        CANDIDATE[cell] = "rec"
for cell in ("c2025", "c1992", "c287", "c2023", "x1", "x3", "d2", "d3", "d5", "d6", "d7", "d8"):
    CANDIDATE.setdefault(cell, "fill_lean")
for cell in ("r2", "r3", "r4"):
    CANDIDATE.setdefault(cell, "nostraddle")

cells = sorted({c for c, _ in runs}, key=lambda c: (c[0], int(c[1:]) if c[1:].isdigit() else 0))
print(
    f"{'cell':6s} {'shape':22s} {'regime':16s} {'op s1':>8s} {'candidate':>12s} {'cand ns':>8s} {'stock':>8s} {'cand/op':>8s} {'cand/stock':>10s}"
)
for cell in cells:
    med = {v: statistics.median(ns) for (c, v), ns in runs.items() if c == cell}
    shape, regime = meta.get(cell, ("?", "?"))
    s1 = med.get("s1")
    cand = CANDIDATE.get(cell, "s1")
    cn = med.get(cand, s1)
    stock = med.get("stock")
    f = lambda x: f"{x:8.0f}" if x is not None else f"{'-':>8s}"
    ratio = lambda a, b: f"{a / b:8.3f}" if a and b else f"{'-':>8s}"
    print(
        f"{cell:6s} {str(shape):22s} {regime[:16]:16s} {f(s1)} {cand:>12s} {f(cn)} {f(stock)} {ratio(cn, s1)} {ratio(cn, stock):>10s}"
    )
    if show_all:
        for v, ns in sorted(med.items(), key=lambda kv: kv[1]):
            print(f"{'':48s} {v:>14s} {ns:8.0f}  n={len(runs[(cell, v)])}")
