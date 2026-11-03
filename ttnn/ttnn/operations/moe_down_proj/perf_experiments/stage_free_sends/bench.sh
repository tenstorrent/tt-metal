#!/bin/bash
# Usage: MOE_VARIANT=base|sf [SF_*=..] [MOE_CASES=a,b] [MOE_CHECK=0] bench.sh -> "<case> <ns>" per row.
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../../../../../.."
source python_env/bin/activate
unset TT_METAL_DPRINT_CORES
export MOE_CASES=${MOE_CASES:-focus}
export MOE_VARIANT=${MOE_VARIANT:-sf}
log=$(mktemp /tmp/moe_sf_XXXX.log)
PYTHONPATH="$here:$PYTHONPATH" scripts/run_safe_pytest.sh --profile --no-precompile --run-all \
    tests/ttnn/unit_tests/operations/moe_down_proj/probes/test_perf_bench.py -p no:randomly -p moe_sf_plugin -s >"$log" 2>&1
grep -E "^(FAILED|ERROR)|[0-9]+ (passed|failed)|SF blocking|PCC|pcc" "$log" | tail -24
csv=$(grep -o "OPs csv generated at: .*" "$log" | tail -1 | sed "s/OPs csv generated at: //")
[ -f "$csv" ] || { echo "no csv"; echo "log: $log"; exit 1; }
echo "zones: python3 $here/../breakdown_r2/zones.py $(dirname "$csv")/profile_log_device.csv"
echo "variant=$MOE_VARIANT x=${SF_X:-1} w=${SF_W:-1} xd=${SF_XD:-2} wd=${SF_WD:-2} kmax=${SF_KMAX:-8} xis=${SF_X_ISSUE_STEPS:-0} wis=${SF_W_ISSUE_STEPS:-0}"
paste -d' ' <(tr ',' '\n' <<<"$MOE_CASES") <(awk -F',' 'NR>1 && /GenericOp/ {print $19}' "$csv")
echo "log: $log"
