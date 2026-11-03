#!/bin/bash
# Usage: run.sh <label> <defines comma list or ""> [cases=focus] [check=1] [kernel_dir=this dir's kernels]
#   -> "<label> <case> <DEVICE KERNEL DURATION ns>". Whole op via the probes test_perf_bench.py.
# CP_SAVE=<prefix> saves outputs (cp_plugin.py). Kernel dir "op" = the op's own kernels/.
here=$(cd "$(dirname "$0")" && pwd)
label=$1; defs=$2; cases=${3:-focus}; check=${4:-1}; kdir=${5:-$here/kernels}
[ "$kdir" = op ] && kdir=$(cd "$here/../../kernels" && pwd)
cd "$here/../../../../../.."
source python_env/bin/activate
unset TT_METAL_DPRINT_CORES
log=$(mktemp /tmp/moe_cp_XXXX.log)
MOE_CASES=$cases MOE_KERNEL_DIR=$kdir MOE_DEFINES=$defs MOE_CHECK=$check PYTHONPATH="$here:$PYTHONPATH" \
    scripts/run_safe_pytest.sh --profile --no-precompile --run-all \
    tests/ttnn/unit_tests/operations/moe_down_proj/probes/test_perf_bench.py -p no:randomly -p cp_plugin >"$log" 2>&1
grep -E "^(FAILED|ERROR)|[0-9]+ (passed|failed)" "$log" | tail -8 | sed "s/^/$label # /"
csv=$(grep -o "OPs csv generated at: .*" "$log" | tail -1 | sed "s/OPs csv generated at: //")
[ -f "$csv" ] || csv=$(ls -t "$TT_METAL_HOME"/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
paste -d' ' <(tr ',' '\n' <<<"$cases") <(awk -F',' 'NR>1 && /GenericOp/ {print $19}' "$csv") | sed "s/^/$label /"
echo "$label # log: $log"
