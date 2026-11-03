#!/bin/bash
# Usage: [MOE_CASES=a,b] [XP_KDIR=..] [XP_DEFINES=..] [XP_X_DEPTH=n] [MOE_CHECK=0] [ZONES=1] bench.sh -> "<case> <ns>"
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../../../../../.."
source python_env/bin/activate
unset TT_METAL_DPRINT_CORES
export MOE_CASES=${MOE_CASES:-focus}
log=$(mktemp /tmp/xp_bench_XXXX.log)
scripts/run_safe_pytest.sh ${XP_DEV:+--dev} --profile --no-precompile --run-all "$here/test_xpipe.py" --import-mode=prepend -p no:randomly >"$log" 2>&1
grep -E "^(FAILED|ERROR)|[0-9]+ (passed|failed)|hang|HANG|PCC|pcc" "$log" | tail -6
csv=$(grep -o "OPs csv generated at: .*" "$log" | awk "{print \$NF}" | tail -1); [ -z "$csv" ] && { echo "no profiler csv"; tail -20 "$log"; exit 1; }
paste -d' ' <(tr ',' '\n' <<<"$MOE_CASES") <(awk -F',' 'NR>1 && /GenericOp/ {print $19}' "$csv")
echo "log: $log"
echo "csv: $(dirname "$csv")/profile_log_device.csv"
[ -n "$ZONES" ] && python3 "$here/../breakdown_r2/zones.py" "$(dirname "$csv")/profile_log_device.csv"
[ -n "$TIMELINE" ] && python3 "$here/timeline.py" "$(dirname "$csv")/profile_log_device.csv"
exit 0
