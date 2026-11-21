# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize runs/<label>.jsonl: per case / variant status, PCC and error vs torch, bit-identity vs the first variant
run, and the true-ulp histogram of the differing outputs.    python3 summ.py <label> [<label> ...]"""
import json
import sys
from pathlib import Path

for label in sys.argv[1:]:
    for line in open(Path(__file__).parent / "runs" / f"{label}.jsonl"):
        r = json.loads(line)
        b = r.get("blocking") or {}
        print(
            f"{r['case']:>12s} {r['variant']:5s} {r['status']:5s} pcc {r.get('pcc') or 0:.9f} "
            f"mae {r.get('mae_vs_torch') or 0:.7f} bit {r.get('bit_identical_to_first')} "
            f"ndiff {r.get('n_diff_vs_first')}/{r.get('n_out', '-')} ulp {r.get('true_ulp_hist', '')} "
            f"maxulp {r.get('max_true_ulp', '-')} maxabs {r.get('max_abs_diff_vs_first', '-')} "
            f"bm {b.get('bm')} bk {b.get('bk')} bn {b.get('bn')} nkb {b.get('k_blocks')} res {b.get('resident')} "
            + (r.get("error", "")[:160] if r["status"] != "ok" else "")
        )
