# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only (torch, no ttnn): bit-level diff of runs/out_<a>_<id>.pt vs runs/out_<b>_<id>.pt, plus the precision
records of both copies from runs/precision.jsonl (last record per copy and case).

    python3 compare_outputs.py op_base op_cand
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(a, b):
    import torch

    recs = {}
    for line in open(HERE / "runs" / "precision.jsonl"):
        r = json.loads(line)
        recs[(r["copy"], r["case"])] = r
    cases = sorted({c for (_, c) in recs})
    for case in cases:
        pa, pb = HERE / "runs" / f"out_{a}_{case}.pt", HERE / "runs" / f"out_{b}_{case}.pt"
        line = f"{case:16s}"
        for cp in (a, b):
            r = recs.get((cp, case))
            if r:
                line += f" | {cp}: pcc {r['pcc']:.6f} mae {r['mae']:.5f} max {r['max_err']:.3f}"
                if "rel_rms" in r:
                    line += f" rrms {r['rel_rms']:.5f}"
        if pa.exists() and pb.exists():
            x, y = torch.load(pa), torch.load(pb)
            diff = (x != y).sum().item()
            line += f" | differ {diff}/{x.numel()} ({100.0 * diff / x.numel():.3f}%), max |a-b| {(x - y).abs().max().item():.4f}"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
