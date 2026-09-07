#!/usr/bin/env bash
set -u
R=/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal
D=$R/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/passb_op_count
cd "$R"
timeout "${RMS_TIMEOUT:-560}" scripts/tt-probe.sh rms_norm_ttnn <<PY 2>&1 | grep -E "RESULT|Traceback|Error" | head -120
import sys
sys.path.insert(0, "$D")
import correctness
correctness.main()
PY
