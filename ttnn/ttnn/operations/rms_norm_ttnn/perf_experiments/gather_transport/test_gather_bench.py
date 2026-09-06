# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Driver for the `gather_transport` isolated bake-off.  Correctness is the ONLY
# pass/fail; perf is printed, never asserted.
#
#   RMS_GRIDS="7x4,8x1"      core grids to sweep (G = w*h, root = logical (0,0))
#   RMS_VARIANTS="col_2x1024,row_2x64"
#   RMS_TRIALS=5

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gather_bench as B

REL_GATE = 5e-2  # the fold runs at fp32_dest_acc_en=False (the op's config): DEST is 16-bit


def _grids():
    spec = os.environ.get("RMS_GRIDS", "7x4")
    out = []
    for tok in spec.split(","):
        w, h = tok.lower().split("x")
        out.append((int(w), int(h)))
    return out


def test_gather_transport_sweep():
    # the two MEASURED-HANG variants are excluded from the default set on purpose
    default = [v for v in B.VARIANTS if "altnoc" not in v and "xb_dest" not in v and "xdest" not in v]
    variants = os.environ.get("RMS_VARIANTS", ",".join(default)).split(",")
    trials = int(os.environ.get("RMS_TRIALS", "5"))
    res = B.sweep(_grids(), variants, trials=trials)
    # `sem_only` is the transport FLOOR probe -- it deliberately moves no payload, so it is
    # not a candidate and is not gated.  Every other variant must reproduce the sum.
    bad = [
        f"G={g}/{v} relerr {r:.3e} pcc {p:.6f}"
        for (g, v), (_, r, p) in res.items()
        if v != "sem_only" and (r > REL_GATE or p < 0.999)
    ]
    assert not bad, "correctness gate failed: " + "; ".join(bad)
