# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join transport_manifest.jsonl (launch order) with the profiler's device kernel ns.

    python3 <this dir>/read_transport.py [ops_perf_results_*.csv]

Every manifest row is exactly ONE ttnn.generic_op launch, in order, so the join is positional.
The first row is the throwaway warm-up program and is skipped, and an `l1_oom` row consumed NO
launch, so the join skips those too.
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
    man = [json.loads(line) for line in open(HERE / "transport_manifest.jsonl")]
    launched = [m for m in man if not m.get("l1_oom")]
    if len(rows) != len(launched):
        print(f"WARNING: {len(rows)} launches vs {len(launched)} launching manifest rows; joining the common prefix")
    it = iter(rows)
    for m in man:
        if m.get("l1_oom"):
            continue
        r = next(it, None)
        if r is None:
            break
        m["ns"] = int(r["DEVICE KERNEL DURATION [ns]"])
        m["cores"] = int(r["CORE COUNT"])

    base = {m["config"]: m["ns"] for m in man if m["variant"] == "flat" and "ns" in m}
    cur = None
    for m in man:
        if m["config"] == "warmup":
            continue
        if m["config"] != cur:
            cur = m["config"]
            print(
                f"\n== {cur}  G={m['group_size']} groups={m['num_groups']} "
                f"BLOCK_ROWS={m['block_rows']} rows={m['num_rows']} rounds={m['num_blocks']}"
            )
        if m.get("l1_oom"):
            print(f"   {m['variant']:16s}   L1_OOM  ({m['l1_bytes']//1024} kB of combine CBs)")
            continue
        sp = (base[m["config"]] / m["ns"]) if m["config"] in base else float("nan")
        per = m["ns"] / m["num_blocks"]
        print(
            f"   {m['variant']:16s} {m['ns']:8d} ns {sp:5.2f}x  {per:8.0f} ns/round  "
            f"pcc={m['pcc']:.7f} rel_rms={m['rel_rms']:.6f}  l1={m['l1_bytes']//1024:4d}kB  "
            f"gather={m['gather_writes']}w/{m['gather_bytes']}B"
        )


if __name__ == "__main__":
    main()
