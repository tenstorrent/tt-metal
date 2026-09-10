#!/usr/bin/env bash
# passb_op_count -- device driver.  FOREGROUND only (device is flock-shared).
#   usage: RMS_NAMES=4 RMS_VARIANTS=base,swap ./run.sh
set -u
R=/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal
D=$R/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/passb_op_count
cd "$R"
timeout "${RMS_TIMEOUT:-1500}" scripts/tt-probe.sh rms_norm_ttnn <<PY 2>&1 | grep -E "RESULT|ERROR|error:|Always|FATAL" | head -200
import sys
sys.path.insert(0, "$D")
import bench
bench.main()
PY
