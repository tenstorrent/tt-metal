# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fold a Tracy per-op CSV back onto a plan_policy manifest and print the table.

    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/plan_policy/pp_report.py \
        <ops_perf_results.csv> [manifest.json]
"""

import csv
import json
import sys
from pathlib import Path

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def report(csv_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    with open(csv_path) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("OP CODE") == "GenericOpDeviceOperation"]
    out, i = [], 0
    for arm in manifest:
        i += arm["calls"] - arm["profiled"]
        window = rows[i : i + arm["profiled"]]
        i += arm["profiled"]
        vals = sorted(float(r[_DURATION_KEY]) for r in window if r.get(_DURATION_KEY))
        out.append(
            dict(
                label=arm["label"],
                cell=arm["cell"],
                ns=(vals[len(vals) // 2] if vals else None),
                lo=(vals[0] if vals else None),
                hi=(vals[-1] if vals else None),
                pcc=arm.get("pcc"),
                levers=arm["levers"],
            )
        )
    print(f"csv rows (GenericOp): {len(rows)}   manifest arms: {len(manifest)}   consumed: {i}")
    cur = None
    for r in out:
        if r["cell"] != cur:
            cur = r["cell"]
            base = next((x for x in out if x["cell"] == cur and x["label"].endswith("policy")), None)
            print(f"\n--- {cur} (policy = {base['ns'] if base else '?'} ns) ---")
        rel = f"{r['ns']/base['ns']:.3f}x" if base and base["ns"] and r["ns"] else ""
        print(f"  {r['label']:<28} {r['ns']:>10.0f} ns  [{r['lo']:.0f}-{r['hi']:.0f}]  {rel:<8} pcc={r['pcc']:.6f}")
    return out


if __name__ == "__main__":
    csv_path = sys.argv[1]
    manifest = sys.argv[2] if len(sys.argv) > 2 else "generated/rms_norm_plan_policy/manifest_groups.json"
    report(csv_path, manifest)
