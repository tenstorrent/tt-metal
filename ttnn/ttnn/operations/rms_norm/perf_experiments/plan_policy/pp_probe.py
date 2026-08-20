# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the plan POLICY picks, per guard-set cell — host-side, no dispatch.

Dumps, for every cell, the policy's own candidate set (spied out of
`_split_cost`), the model score of each G, and the pick.  This is the "what does
the policy think" half; `test_plan_bench.py` measures the "what is actually
fastest" half.

    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/rms_norm/test_pp_plan_policy.py -k probe -s
"""

import json

from ttnn.operations.rms_norm.perf_experiments.plan_policy import pp_common as pp


def run(device):
    grid = device.compute_with_storage_grid_size()
    print(f"\nGRID {grid.x} x {grid.y} = {grid.x * grid.y} cores")
    out = {}
    for cell in pp.CELLS:
        rows, plan, pick = pp.candidate_table(device, cell)
        out[cell] = dict(pick=pick, cands=rows)
        print(f"\n=== {cell}  Wt={pick['Wt']} Rt={pick['Rt']} -> PICK G={pick['G']} ===")
        print(f"    {pick}")
        best = min(r["cost"] for r in rows)
        for r in sorted(rows, key=lambda r: r["g"]):
            mark = "*" if r["g"] == pick["G"] else " "
            print(
                f"  {mark} G={r['g']:<4} cost={r['cost']:>10.1f} ({r['cost']/best:.3f}x)  "
                f"Wt_core={r['Wt_core']:<4} regime={r['regime']} bht={r['block_ht']} wr={r['wr']} "
                f"in={r['in_depth']} out={r['out_depth']} gd={r['gamma_depth']} "
                f"nrb={r['num_row_blocks']} groups={r['num_groups']}/{r['groups_used']}"
            )
    pp.PLANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pp.PLANS_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nplans -> {pp.PLANS_PATH}")
