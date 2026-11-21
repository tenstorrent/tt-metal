# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize runs/<label>.jsonl: per case and variant, status, PCC (bands and whole output), bit identity / ulp vs the
first variant, worst error in reference ulps, and the plan. Usage: python3 -I summ.py <label> [<label> ...]"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
for label in sys.argv[1:]:
    for line in open(HERE / "runs" / f"{label}.jsonl"):
        r = json.loads(line)
        if r.get("status") != "ok":
            print(r["case"], r["variant"], r.get("status"), r.get("error"))
            continue
        p = r.get("plan") or {}
        b = r.get("blocking") or {}
        print(
            f"{r['case']:>4} {r['variant']:6s} pcc {r['pcc_full']:.7f} errulp {r['max_err_in_ref_ulps']:.3f} "
            f"ident {int(r['bit_identical_to_first'])} ndiff {r['n_diff_vs_first']} ({r['frac_diff_vs_first']*100:.3f}%) "
            f"maxulp {r['max_ulp_vs_first']} sig>1 {r['ulp_gt1_significant']} flips {r['sign_flips_vs_first']} "
            f"| bm {b.get('bm')} bk {b.get('bk')} nkb {b.get('nkb')} ring {b.get('ring')} khr {b.get('khr')} "
            f"| rs rows {p.get('rows')} lag {p.get('lag')} out {p.get('out_rg')} sg {p.get('store_groups')} "
            f"bias {p.get('bias_mode')}"
        )
