# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join last_run_manifest.jsonl (launch order) with the profiler's device kernel ns.

    python3 <this dir>/read_results.py [ops_perf_results_*.csv] [out.json]

Every manifest row is exactly ONE ttnn.generic_op launch, in order, so the join is
positional.  The first row is the throwaway warm-up program and is skipped.
Deliberately NOT importable as a package module -- run it directly.
"""
import csv
import glob
import json
import os
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).parent


def main():
    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime)[-1]
    )
    rows = [r for r in csv.DictReader(open(csv_path)) if r["OP CODE"] == "GenericOpDeviceOperation"]
    man = [json.loads(l) for l in open(HERE / "last_run_manifest.jsonl")]
    if len(rows) != len(man):
        print(f"WARNING: {len(rows)} launches vs {len(man)} manifest rows; joining the common prefix")
    recs = []
    for m, r in zip(man, rows):
        m["ns"] = int(r["DEVICE KERNEL DURATION [ns]"])
        m["cores"] = int(r["CORE COUNT"])
        recs.append(m)
    recs = [m for m in recs if m["config"] != "warmup"]

    # median over repeats of the same (config, variant)
    agg = {}
    for m in recs:
        agg.setdefault((m["config"], m["variant"]), []).append(m)
    base = {}
    for (cid, var), ms in agg.items():
        if var == "flat":
            base[cid] = statistics.median(x["ns"] for x in ms)

    cur = None
    seen = set()
    for m in recs:
        key = (m["config"], m["variant"])
        if key in seen:
            continue
        seen.add(key)
        ms = agg[key]
        ns = statistics.median(x["ns"] for x in ms)
        if m["config"] != cur:
            cur = m["config"]
            print(
                f"\n== {cur}  cores={m['cores']} G={m['group_size']} groups={m['num_groups']} "
                f"BLOCK_ROWS={m['block_rows']} rows={m['num_rows']}"
            )
        sp = (base[m["config"]] / ns) if m["config"] in base else float("nan")
        n = f" (n={len(ms)}: {[x['ns'] for x in ms]})" if len(ms) > 1 else ""
        print(
            f"   {m['variant']:16s} {int(ns):8d} ns  {sp:5.2f}x  pcc={m['pcc']:.6f}  "
            f"rel_rms={m['rel_rms']:.5f}  l1={m['l1_bytes']//1024:4d}kB  sems={m['sems']}{n}"
        )
    if len(sys.argv) > 2:
        out = pathlib.Path(sys.argv[2])
        out.write_text(json.dumps(recs, indent=1))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
