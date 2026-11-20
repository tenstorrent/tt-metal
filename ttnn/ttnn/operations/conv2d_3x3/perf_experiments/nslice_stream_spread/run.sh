#!/usr/bin/env bash
# One profiled session of test_nss.py into this dir's own profiler folder (siblings share the default one).
#   NSS_E_CASES=... NSS_E_VARIANTS=... [NSS_E_ORDER=alt] [NSS_E_NOCHECK=1] [NSS_E_NZ=1] [NSS_E_DEV=1] [NSS_E_NOPROF=1]
#   run.sh <tag>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"
export NSS_E_LOG="${TAG}.jsonl"
export TT_METAL_PROFILER_DIR="$HERE/prof/$TAG"
rm -rf "$TT_METAL_PROFILER_DIR"
mkdir -p "$HERE/runs"
python3 "$HERE/mk_twins.py" > /dev/null || exit 1
python3 "$HERE/zone_hash_check.py" || exit 1
cd "$REPO"
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source python_env/bin/activate
LOG="$HERE/runs/${TAG}.log"
FLAGS="--run-all"
[[ "${NSS_E_DEV:-0}" == "1" ]] && FLAGS="--dev $FLAGS"
[[ "${NSS_E_DEV:-0}" != "1" && "${NSS_E_NOPROF:-0}" != "1" ]] && FLAGS="--profile $FLAGS"
# The shared device lock: give up (exit 3) if it is not acquired within NSS_E_LOCK_WAIT seconds (default 240), so a
# caller with a bounded wall clock can retry; never stopped once it holds the device.
scripts/run_safe_pytest.sh $FLAGS "$HERE/test_nss.py" --import-mode=prepend -q > "$LOG" 2>&1 &
PID=$!
START=$(date +%s)
while kill -0 $PID 2>/dev/null; do
    if ! grep -q "Device lock acquired" "$LOG"; then
        if (( $(date +%s) - START > ${NSS_E_LOCK_WAIT:-240} )); then
            pkill -P $PID 2>/dev/null
            kill $PID 2>/dev/null
            echo "LOCK_TIMEOUT after $(( $(date +%s) - START ))s (not started)"
            exit 3
        fi
    fi
    sleep 3
done
wait $PID
grep -E "PASS|FAIL|Error|error:|SAFE_PYTEST|passed|failed|skipped|assert|Timeout|hang|HANG|triage" "$LOG" | \
    grep -v "^PRECOMPILE\|waiting for device" | tail -40
[[ "$FLAGS" != *--profile* ]] && exit 0
CSV="$TT_METAL_PROFILER_DIR/.logs/cpp_device_perf_report.csv"
[[ -f "$CSV" ]] || CSV=$(ls -t "$TT_METAL_PROFILER_DIR"/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
echo "CSV: $CSV"
[[ -n "$CSV" ]] && python3 "$HERE/report.py" "$CSV" "$HERE/runs/${TAG}.jsonl" --append "$HERE/results.jsonl" --tag "$TAG"
[[ -n "$CSV" ]] && cp "$CSV" "$HERE/runs/${TAG}_perf.csv"
[[ -f "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" ]] && \
    gzip -c "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" > "$HERE/runs/${TAG}_device.csv.gz"
rm -rf "$TT_METAL_PROFILER_DIR"
