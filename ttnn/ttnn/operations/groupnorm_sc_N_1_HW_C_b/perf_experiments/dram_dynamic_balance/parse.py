#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join runs.jsonl with the ops CSV (GenericOp rows in order) -> case x variant table of device ns.
usage: parse.py <ops_perf_results.csv> [runs.jsonl] [--save tag]"""
import csv, json, sys
from pathlib import Path
from collections import OrderedDict

HERE = Path(__file__).parent
csvp = sys.argv[1]
runsp = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else HERE / "runs.jsonl"
runs = [json.loads(l) for l in open(runsp) if l.strip()]
rows = [r for r in csv.DictReader(open(csvp)) if r.get("DEVICE KERNEL DURATION [ns]", "").strip()]
assert len(rows) == len(runs), (len(rows), len(runs))
tab = OrderedDict()
variants = []
for run, row in zip(runs, rows):
    run["ns"] = int(float(row["DEVICE KERNEL DURATION [ns]"]))
    tab.setdefault(run["case"], OrderedDict())[run["variant"]] = run
    if run["variant"] not in variants:
        variants.append(run["variant"])
short = [v.split("/")[-1][-14:] for v in variants]
print(f"{'case':28s}" + "".join(f"{s:>16s}" for s in short))
for case, d in tab.items():
    base = d.get(variants[0], {}).get("ns")
    cells = []
    for v in variants:
        r = d.get(v)
        if r is None:
            cells.append(f"{'-':>16s}")
            continue
        pct = f"{(r['ns'] / base - 1) * 100:+.1f}%" if base and v != variants[0] else ""
        cells.append(f"{r['ns']:>9,d}{pct:>7s}")
    print(f"{case:28s}" + "".join(cells))
    for v in variants:
        r = d.get(v)
        if r and "pcc" in r:
            pass
if "--save" in sys.argv:
    tag = sys.argv[sys.argv.index("--save") + 1]
    json.dump(runs, open(HERE / f"results_{tag}.json", "w"), indent=1)
