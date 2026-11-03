#!/bin/bash
# Usage: [MOE_CASES=a,b] [MOE_KERNEL_DIR=..] [MOE_DEFINES=..] [MOE_CHECK=0] bench.sh  -> "<case> <ns>" per row.
# Profiles test_perf_bench.py (one fresh-cache run per case) and prints DEVICE KERNEL DURATION [ns].
cd "$(dirname "$0")/../../../../../.."
source python_env/bin/activate
unset TT_METAL_DPRINT_CORES
export MOE_CASES=${MOE_CASES:-focus}
log=$(mktemp /tmp/moe_bench_XXXX.log)
scripts/run_safe_pytest.sh --profile --no-precompile --run-all \
    tests/ttnn/unit_tests/operations/moe_down_proj/probes/test_perf_bench.py -p no:randomly >"$log" 2>&1
grep -E "^(FAILED|ERROR)|[0-9]+ (passed|failed)" "$log" | tail -5
csv=$(ls -t "$TT_METAL_HOME"/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
paste -d' ' <(tr ',' '\n' <<<"$MOE_CASES") <(awk -F',' 'NR>1 && /GenericOp/ {print $19}' "$csv")
echo "log: $log"
