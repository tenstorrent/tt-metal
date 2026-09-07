# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Entry point for the `per_channel_boot_overlap` bake-off.  Run it through the
# device wrapper (flock + hang detection + reset), NEVER bare python3:
#
#   RMS_VARIANTS=base,split RMS_NAMES=FOCUS_2304 \
#     scripts/tt-probe.sh rms_norm_ttnn <<'PYEOF'
#   exec(open("ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/per_channel_boot_overlap/run_bench.py").read())
#   PYEOF
#
# A pytest file cannot live here: pytest.ini forces --import-mode=importlib, so a
# test under ttnn/ttnn/... is imported as `ttnn.ttnn....` and re-registers every
# ttnn C++ op.
import os
import sys
from pathlib import Path

HERE = Path("ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/per_channel_boot_overlap").resolve()
sys.path.insert(0, str(HERE))

import bench as B

PCC_GATE = 0.9995

labels = os.environ.get("RMS_VARIANTS", "base,split").split(",")
names = os.environ.get("RMS_NAMES", "").split(",") if os.environ.get("RMS_NAMES") else list(B.CASES)
res = B.sweep(labels, names=names)

bad = []
for (name, label), samples in res.items():
    p = min(s[1] for s in samples)
    if p < PCC_GATE and "ablate" not in label:
        bad.append(f"{name}/{label} pcc {p:.6f}")
if bad:
    print("RESULT CORRECTNESS-FAIL " + "; ".join(bad))
    raise SystemExit(1)
print("RESULT CORRECTNESS-OK (all variants pcc >= %.4f)" % PCC_GATE)
