# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Entry point for the `reader_boot_order` bake-off.  Run it through the device
# wrapper (flock + hang detection + reset), never bare python3:
#
#   RMS_VARIANTS=base,a_pub_first RMS_NAMES=F_w7168_28c \
#     scripts/tt-probe.sh rms_norm_ttnn <<'PYEOF'
#   exec(open("ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/reader_boot_order/run_bench.py").read())
#   PYEOF
#
# A pytest file cannot live here: pytest.ini forces --import-mode=importlib, so a
# test under ttnn/ttnn/... is imported as `ttnn.ttnn....` and re-registers every
# ttnn C++ op ("Operation with name \"bernoulli\" is already registered").
import os
import sys
from pathlib import Path

HERE = Path("ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/reader_boot_order").resolve()
sys.path.insert(0, str(HERE))

import bench_boot_order as B

PCC_GATE = 0.9995

labels = os.environ.get("RMS_VARIANTS", "base,a_pub_first,b_scaler_pub,c_split").split(",")
names = os.environ.get("RMS_NAMES", "").split(",") if os.environ.get("RMS_NAMES") else list(B.CASES)
res = B.sweep(labels, names=names)

bad = []
for (name, label), samples in res.items():
    p = min(s[1] for s in samples)
    if p < PCC_GATE:
        bad.append(f"{name}/{label} pcc {p:.6f}")
if bad:
    print("RESULT CORRECTNESS-FAIL " + "; ".join(bad))
    raise SystemExit(1)
print("RESULT CORRECTNESS-OK (all variants pcc >= %.4f)" % PCC_GATE)
