# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Entry point for the dual_noc_write DM ROOFLINE INSTRUMENT.  Run through the
# device wrapper (flock + hang detection + reset), never bare python3:
#
#   DM_SPECS=base,swap_noc DM_SHAPE=8192x2304 \
#     scripts/tt-probe.sh rms_norm_ttnn <<'PYEOF'
#   exec(open("ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/dual_noc_write/run_dm_bench.py").read())
#   PYEOF
#
# A spec is `config[:flags]` where flags are any of  ar (ablate read),
# aw (ablate write), wf (write-flush instead of the per-block write barrier),
# dN (CB depth N).
import os
import sys
from pathlib import Path

HERE = Path("ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/dual_noc_write").resolve()
sys.path.insert(0, str(HERE))

import ttnn
import dm_bench as B

shape_s = os.environ.get("DM_SHAPE", "8192x2304")
H, W = (int(v) for v in shape_s.split("x"))
SHAPE = [1, 1, H, W]
TRIALS = int(os.environ.get("DM_TRIALS", "3"))

specs = os.environ.get("DM_SPECS", "base,base:ar,base:aw,base:ar+aw").split(",")


def parse(spec):
    parts = spec.split(":")
    cfg = parts[0]
    flags = set(parts[1].split("+")) if len(parts) > 1 and parts[1] else set()
    kw = dict(ablate_read="ar" in flags, ablate_write="aw" in flags, write_flush="wf" in flags)
    depth = 2
    for f in flags:
        if f.startswith("d") and f[1:].isdigit():
            depth = int(f[1:])
    kw["depth"] = depth
    nbanks = 0
    for f in flags:
        if f.startswith("bank") and f[4:].isdigit():
            nbanks = int(f[4:])
    kw["nbanks"] = nbanks
    alt = 0
    for f in flags:
        if f.startswith("alt") and f[3:].isdigit():
            alt = int(f[3:])
    kw["alt_tail"] = alt
    kw["dyn"] = ("dyn" in flags) or alt > 0
    return cfg, kw


device = ttnn.open_device(device_id=0)
try:
    Rt = (H + 31) // 32
    WT = (W + 31) // 32
    total_mb = Rt * WT * 2048 / 1e6
    print(f"RESULT shape={SHAPE} Rt={Rt} WT={WT} payload={total_mb:.2f} MB each way")
    rows = []
    for spec in specs:
        cfg, kw = parse(spec)
        ns, exact, best = B.measure(device, SHAPE, cfg, trials=TRIALS, **kw)
        moved = total_mb * ((0 if kw["ablate_read"] else 1) + (0 if kw["ablate_write"] else 1))
        gbs = (moved * 1e6) / ns if ns and ns == ns else 0.0
        rows.append((spec, ns, best, exact, gbs))
        print(f"RESULT {spec:24s} median_ns={ns:12.0f} best_ns={best:12.0f} exact={exact}  agg_GB/s={gbs:7.1f}")
    print("RESULT ---")
    ref = rows[0][1]
    for spec, ns, best, exact, gbs in rows:
        print(f"RESULT SPEEDUP {spec:24s} {ref/ns:6.3f}x vs {rows[0][0]}")
finally:
    ttnn.close_device(device)
