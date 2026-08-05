# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join last_run_manifest.jsonl (launch order) with the profiler's device kernel ns.

    python3 <this dir>/read_results.py [ops_perf_results_*.csv]

Every manifest row is exactly ONE ttnn.generic_op launch, in order, so the join is
positional.  The first row is the throwaway warm-up program and is skipped.
Deliberately NOT importable as a package module -- run it directly.
"""
import csv
import glob
import json
import os
import pathlib
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
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    base = {m["config"]: m["ns"] for m in recs if m["variant"] == "flat"}
    cur = None
    for m in recs:
        if m["config"] == "warmup":
            continue
        if m["config"] != cur:
            cur = m["config"]
            print(
                f"\n== {cur}  cores={m['cores']} G={m['group_size']} groups={m['num_groups']} "
                f"BLOCK_ROWS={m['block_rows']} rows={m['num_rows']}"
            )
        # `flat` can be absent: at BLOCK_ROWS = 32 the flat root's GROUP_SIZE * BLOCK_ROWS
        # gather ring does not fit in L1 at all, so there is no baseline to divide by.
        sp = (base[m["config"]] / m["ns"]) if m["config"] in base else float("nan")
        print(
            f"   {m['variant']:16s} {m['ns']:8d} ns  {sp:5.2f}x  pcc={m['pcc']:.6f}  "
            f"rel_rms={m['rel_rms']:.5f}  l1={m['l1_bytes']//1024:4d}kB  sems={m['sems']}"
        )
    if out is not None:
        out.write_text(json.dumps([m for m in recs if m["config"] != "warmup"], indent=1))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
