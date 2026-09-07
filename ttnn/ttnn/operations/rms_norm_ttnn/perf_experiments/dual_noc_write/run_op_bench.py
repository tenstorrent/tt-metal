# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
#   RMS_VARIANTS=base,kbase,dyn,alt RMS_NAMES=FOCUS_8192x2304 \
#     scripts/tt-probe.sh rms_norm_ttnn <<'PYEOF'
#   exec(open("ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/dual_noc_write/run_op_bench.py").read())
#   PYEOF
import os
import sys
from pathlib import Path

HERE = Path("ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/dual_noc_write").resolve()
sys.path.insert(0, str(HERE))

import bench_op as B

PCC_GATE = 0.9995
labels = os.environ.get("RMS_VARIANTS", "base,kbase,dyn,alt").split(",")
names = os.environ.get("RMS_NAMES", "").split(",") if os.environ.get("RMS_NAMES") else list(B.CASES)
res = B.sweep(labels, names)

bad = [f"{n}/{l} pcc {p:.6f}" for (n, l), (_, p) in res.items() if p < PCC_GATE]
if bad:
    print("RESULT CORRECTNESS-FAIL " + "; ".join(bad))
    raise SystemExit(1)
print("RESULT CORRECTNESS-OK (all variants pcc >= %.4f)" % PCC_GATE)
