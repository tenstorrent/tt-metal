#!/bin/bash
# Usage: MOE_VARIANT=base|tail [MOE_CASES=a,b] [MOE_TAIL_SPLIT=2] [MOE_CHECK=0] bench.sh -> "<case> <ns>" per row.
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../../../../../.."
source python_env/bin/activate
unset TT_METAL_DPRINT_CORES
export MOE_CASES=${MOE_CASES:-focus}
export MOE_VARIANT=${MOE_VARIANT:-tail}
[ "$MOE_VARIANT" = base ] && export MOE_KERNEL_DIR="$here/kernels_base"
log=$(mktemp /tmp/moe_tail_XXXX.log)
PYTHONPATH="$here:$PYTHONPATH" scripts/run_safe_pytest.sh --profile --no-precompile --run-all tests/ttnn/unit_tests/operations/moe_down_proj/probes/test_perf_bench.py -p no:randomly -p moe_tail_plugin >"$log" 2>&1
grep -E "^(FAILED|ERROR)|[0-9]+ (passed|failed)|TAIL blocking" "$log" | tail -12
csv=$(grep -o "OPs csv generated at: .*" "$log" | tail -1 | sed "s/OPs csv generated at: //")
[ -f "$csv" ] || { echo "no csv"; exit 1; }
echo "zones: python3 $here/../breakdown/zones.py $(dirname "$csv")/profile_log_device.csv"
echo "variant=$MOE_VARIANT split=${MOE_TAIL_SPLIT:-2} rot=${MOE_STORE_ROT:-0} cmdbufs=${MOE_STORE_CMDBUFS:-1}"
paste -d' ' <(tr ',' '\n' <<<"$MOE_CASES") <(awk -F',' 'NR>1 && /GenericOp/ {print $19}' "$csv")
echo "log: $log"
