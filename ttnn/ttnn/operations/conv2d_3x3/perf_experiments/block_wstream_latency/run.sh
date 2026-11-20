#!/usr/bin/env bash
# One profiled session of test_wsl.py into this dir's own profiler folder (siblings share the default one).
#   WSL_CASES=... WSL_VARIANTS=... [WSL_ORDER=alt] [WSL_REPEAT=n] [WSL_NOCHECK=1] [WSL_NZ=1] [WSL_DEV=1] [WSL_NOPROF=1]
#   run.sh <tag>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"
export WSL_LOG="${TAG}.jsonl"
export TT_METAL_PROFILER_DIR="$HERE/prof/$TAG"
rm -rf "$TT_METAL_PROFILER_DIR" "$HERE/runs/${TAG}.jsonl"
mkdir -p "$HERE/runs"
python3 "$HERE/zone_hash_check.py" || exit 1
cd "$REPO"
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source python_env/bin/activate
LOG="$HERE/runs/${TAG}.log"
FLAGS="--run-all"
[[ "${WSL_DEV:-0}" == "1" ]] && FLAGS="--dev $FLAGS"
[[ "${WSL_DEV:-0}" != "1" && "${WSL_NOPROF:-0}" != "1" ]] && FLAGS="--profile $FLAGS"
# The shared device lock: give up (exit 3) if it is not acquired within WSL_LOCK_WAIT seconds (default 360), so the
# caller can retry; never stopped once it holds the device.
scripts/run_safe_pytest.sh $FLAGS "$HERE/test_wsl.py" --import-mode=prepend -q > "$LOG" 2>&1 &
PID=$!
START=$(date +%s)
while kill -0 $PID 2>/dev/null; do
    if ! grep -q "Device lock acquired" "$LOG"; then
        if (( $(date +%s) - START > ${WSL_LOCK_WAIT:-360} )); then
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
echo "CSV: $CSV"
[[ -n "$CSV" ]] && python3 "$HERE/report.py" "$CSV" "$HERE/runs/${TAG}.jsonl" --append "$HERE/results.jsonl" --tag "$TAG"
[[ -n "$CSV" ]] && cp "$CSV" "$HERE/runs/${TAG}_ops_perf.csv"
[[ -f "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" ]] && \
    gzip -c "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" > "$HERE/runs/${TAG}_device.csv.gz"
rm -rf "$TT_METAL_PROFILER_DIR"
