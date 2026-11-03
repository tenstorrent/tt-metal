#!/bin/bash
# Usage: [MOE_CASES=a,b] [EA_KDIR=..] [EA_DEFINES=EARLY_ACK_X,EARLY_ACK_W] [MOE_CHECK=0] bench.sh -> "<case> <ns>"
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../../../../../.."
source python_env/bin/activate
unset TT_METAL_DPRINT_CORES
export MOE_CASES=${MOE_CASES:-focus}
log=$(mktemp /tmp/ea_bench_XXXX.log)
scripts/run_safe_pytest.sh ${EA_DEV:+--dev} --profile --no-precompile --run-all "$here/test_early_ack.py" --import-mode=prepend -p no:randomly >"$log" 2>&1
grep -E "^(FAILED|ERROR)|[0-9]+ (passed|failed)|hang|HANG" "$log" | tail -5
csv=$(grep -o "OPs csv generated at: .*" "$log" | awk "{print \$NF}" | tail -1); [ -z "$csv" ] && { echo "no profiler csv"; tail -20 "$log"; exit 1; }
paste -d' ' <(tr ',' '\n' <<<"$MOE_CASES") <(awk -F',' 'NR>1 && /GenericOp/ {print $19}' "$csv")
echo "log: $log"
[ -n "$ZONES" ] && python3 "$here/../breakdown/zones.py" "$(dirname "$csv")/profile_log_device.csv"
