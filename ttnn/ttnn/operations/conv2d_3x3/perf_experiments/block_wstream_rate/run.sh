#!/usr/bin/env bash
# One profiled session of test_bwr.py into this dir's own profiler folder (siblings share the default one).
#   BWR_CASES=... BWR_VARIANTS=... [BWR_ORDER=alt] [BWR_NOCHECK=1] [BWR_NZ=1 | BWR_ABL=1 (sets CONV2D_3X3_ABLATE=scatter)]
#   [BWR_DEV=1] [BWR_NOPROF=1] [BWR_LOCK_WAIT=s] run.sh <tag>
# Gives up (exit 3) if the shared device lock is not acquired within BWR_LOCK_WAIT seconds (default 1800); it is only
# ever stopped while waiting for the lock, never once it holds the device.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"
export BWR_LOG="${TAG}.jsonl"
export TT_METAL_PROFILER_DIR="$HERE/prof/$TAG"
[[ "${BWR_ABL:-0}" == "1" ]] && export CONV2D_3X3_ABLATE=scatter
rm -rf "$TT_METAL_PROFILER_DIR" "$HERE/runs/${TAG}.jsonl"
python3 "$HERE/mk_twins.py" > /dev/null || exit 1
python3 "$HERE/zone_hash_check.py" || exit 1
cd "$REPO"
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source python_env/bin/activate
LOG="$HERE/runs/${TAG}.log"
FLAGS="--run-all"
[[ "${BWR_DEV:-0}" == "1" ]] && FLAGS="--dev $FLAGS"
[[ "${BWR_DEV:-0}" != "1" && "${BWR_NOPROF:-0}" != "1" ]] && FLAGS="--profile $FLAGS"
scripts/run_safe_pytest.sh $FLAGS "$HERE/test_bwr.py" --import-mode=prepend -q > "$LOG" 2>&1 &
PID=$!
START=$(date +%s)
while kill -0 $PID 2>/dev/null; do
    if ! grep -q "Device lock acquired" "$LOG"; then
        if (( $(date +%s) - START > ${BWR_LOCK_WAIT:-1800} )); then
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
CSV=$(ls -t "$TT_METAL_PROFILER_DIR"/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
echo "CSV: $CSV"
[[ -n "$CSV" ]] && python3 "$HERE/report.py" "$CSV" "$HERE/runs/${TAG}.jsonl" --append "$HERE/results.jsonl" --tag "$TAG"
[[ -n "$CSV" ]] && cp "$CSV" "$HERE/runs/${TAG}_ops_perf.csv"
[[ -f "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" ]] && \
    gzip -c "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" > "$HERE/runs/${TAG}_device.csv.gz"
rm -rf "$TT_METAL_PROFILER_DIR"
