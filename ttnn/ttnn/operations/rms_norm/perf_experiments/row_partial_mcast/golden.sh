#!/bin/bash
# golden.sh <tag> [stride]: the golden correctness sweep of the clone, log in runs/<tag>.log
cd /localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal
source python_env/bin/activate
D=ttnn/ttnn/operations/rms_norm/perf_experiments/row_partial_mcast
GR_STRIDE=${2:-1} scripts/run_safe_pytest.sh --run-all $D/golden_row.py --import-mode=prepend -q > $D/runs/$1.log 2>&1
echo "DONE rc=$?" >> $D/runs/$1.log
