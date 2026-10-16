#!/bin/bash
# usage: run_variant.sh <label> <case-id> [MRS_FILL defines]  -> runs traced+profiled, saves report dir copy + summary
# case ids: M640_K2048_N7168_sd-1 | M640_K4096_N6144_sd-1 | M2048_K2048_N4096_sd-2 | M640_K2048_N7168_sd-1_lin
set -u
REPO=/home/malimpic/localdev/2026_10_15/1521_malimpic_matmul-reduce-scatter-eval/clones/matmul_reduce_scatter_run1/tt-metal
D=$REPO/ttnn/ttnn/operations/matmul_reduce_scatter/perf_experiments/entry0_fill
label=$1; case=$2; fill=${3:-}
cd $REPO; source python_env/bin/activate
out=$D/results/${label}__${case}; mkdir -p $out
MRS_PERF_TRACE=1 MRS_FILL="$fill" scripts/run_safe_pytest.sh --profile "$D/test_fill_perf.py::test_fill_case[$case]" -s --rootdir=$D > $out/log.txt 2>&1
csv=$(grep "PROFILER CSV" $out/log.txt | sed 's/.*PROFILER CSV: //')
grep -E "FILL_RESULT|passed|failed|Error|SAFE_PYTEST_RESULT" $out/log.txt | grep -v "^E  " | cut -c1-300
if [ -n "$csv" ]; then
  cp "$csv" $out/ops.csv; cp "$(dirname $csv)/profile_log_device.csv" $out/device.csv
  python $D/extract.py $out/ops.csv ${MRS_MEASURED:-3} | tee $out/latency.txt
fi
