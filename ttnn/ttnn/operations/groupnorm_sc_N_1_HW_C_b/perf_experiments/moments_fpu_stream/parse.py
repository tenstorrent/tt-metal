#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join runs_<tag>.jsonl with the ops CSV (GenericOp rows in order) -> case x variant table of device ns, plus the
precision columns (pcc64 / rms64 vs fp64, sha1 of the output: equal sha = bit-identical output).
usage: parse.py <ops_perf_results.csv> <runs.jsonl> [--save tag]
       parse.py --noprof <runs.jsonl>"""
import csv, json, sys
from pathlib import Path
from collections import OrderedDict

HERE = Path(__file__).parent
noprof = sys.argv[1] == "--noprof"
runsp = Path(sys.argv[2])
runs = [json.loads(l) for l in open(runsp) if l.strip()]
if not noprof:
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("DEVICE KERNEL DURATION [ns]", "").strip()]
    assert len(rows) == len(runs), (len(rows), len(runs))
    for run, row in zip(runs, rows):
        run["ns"] = int(float(row["DEVICE KERNEL DURATION [ns]"]))
tab = OrderedDict()
variants = []
for run in runs:
    key = run["case"] if run.get("dist", "randn") == "randn" else f"{run['case']}[{run['dist']}]"
    tab.setdefault(key, OrderedDict())[run["variant"]] = run
    if run["variant"] not in variants:
        variants.append(run["variant"])
short = [v.split("/")[-1][-16:] for v in variants]
print(f"{'case':28s}" + "".join(f"{s:>18s}" for s in short))
for case, d in tab.items():
    base = d.get(variants[0], {}).get("ns")
    cells = []
    for v in variants:
        r = d.get(v)
        if r is None or "ns" not in r:
            cells.append(f"{'-':>18s}")
            continue
        pct = f"{(r['ns'] / base - 1) * 100:+.1f}%" if base and v != variants[0] else ""
        cells.append(f"{r['ns']:>10,d}{pct:>8s}")
    print(f"{case:28s}" + "".join(cells))
print()
for case, d in tab.items():
    for v, r in d.items():
        if "pcc64" in r:
            same = (
                ""
                if v == variants[0]
                else ("bit-identical" if r.get("sha1") == d[variants[0]].get("sha1") else "DIFFERS")
            )
            print(
                f"  {case:36s} {r.get('dist', ''):7s} {v[-16:]:>16s} pcc64={r['pcc64']:.7f} rms64={r['rms64']:.5f} "
                f"pcc={r['pcc']:.6f} rms={r['rms']:.5f} sha={r['sha1']} {same}"
            )
if "--save" in sys.argv:
    tag = sys.argv[sys.argv.index("--save") + 1]
    json.dump(runs, open(HERE / f"results_{tag}.json", "w"), indent=1)
