#!/bin/bash
# usage: PB_CELLS=a,b PB_VARIANTS=x,y run.sh <tag>   (foreground only)
# Profiled bench run; copies the ops CSV + device log into runs/<tag>.*, prints per-cell base vs variant ns.
set -u
ROOT=/localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal
MY=$ROOT/ttnn/ttnn/operations/rms_norm/perf_experiments/dram_noc_balance/port
cd $ROOT && source python_env/bin/activate
mkdir -p $MY/runs
out=$(scripts/run_safe_pytest.sh --profile $MY/bench_port.py -s --import-mode=prepend 2>&1); csv=$(echo "$out" | sed -n 's/^SAFE_PYTEST: PROFILER CSV: //p')
[ -n "$csv" ] && cp "$csv" $MY/runs/$1.ops.csv; cp generated/profiler/.logs/profile_log_device.csv $MY/runs/$1.dev.csv
echo "$out" > $MY/runs/$1.log
echo "$out" | grep -E "^PB |SAFE_PYTEST_RESULT|Error|FAILED|assert|passed|failed" | grep -v "^PB .* base pcc" | head -80
python3 $MY/table.py $MY/runs/$1.ops.csv $MY/runs/$1.log
