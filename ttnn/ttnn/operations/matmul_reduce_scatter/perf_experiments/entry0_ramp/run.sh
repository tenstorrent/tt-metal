#!/usr/bin/env bash
# usage: run.sh <label> [MRS_CASES list]   (env knobs MRS_* pass through: MRS_RAMP, MRS_RAMP_PKG=op, MRS_PERF_GUARD, ...)
# One fresh-cache traced run under the profiler; prints PLAN / rel_rms / per-case replay latency (max over chips).
set -u
REPO=/home/malimpic/localdev/2026_10_15/1521_malimpic_matmul-reduce-scatter-eval/clones/matmul_reduce_scatter_run1/tt-metal
D=$REPO/ttnn/ttnn/operations/matmul_reduce_scatter/perf_experiments/entry0_ramp
label=$1; export MRS_CASES=${2:-${MRS_CASES:-}}
cd $REPO; source python_env/bin/activate
out=$D/results/$label; mkdir -p $out
MRS_PERF_TRACE=${MRS_PERF_TRACE:-1} scripts/run_safe_pytest.sh --profile "$D/test_ramp_perf.py" -s --rootdir=$D > $out/log.txt 2>&1
grep -E "^PLAN|SAFE_PYTEST_RESULT|passed|failed|error:|Error" $out/log.txt | grep -v "^E  " | cut -c1-330 | sort -u | head -20
csv=$(grep "PROFILER CSV" $out/log.txt | tail -1 | sed 's/.*PROFILER CSV: //')
if [ -n "$csv" ]; then
  cp "$csv" $out/ops.csv; cp "$(dirname $csv)/profile_log_device.csv" $out/device.csv
  python3 tests/ttnn/unit_tests/operations/matmul_reduce_scatter/perf_summary.py $out/ops.csv | tee $out/summary.txt
fi
