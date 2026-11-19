#!/usr/bin/env bash
# One profiled session of test_s2.py into this dir's own profiler folder (siblings share the default one).
#   S2_CASES=... S2_VARIANTS=... [S2_ORDER=alt] [S2_NOCHECK=1] [S2_NZ=1] [S2_DEV=1] [S2_NOPROF=1] run.sh <tag>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"
export S2_LOG="${TAG}.jsonl"
export TT_METAL_PROFILER_DIR="$HERE/prof/$TAG"
rm -rf "$TT_METAL_PROFILER_DIR" "$HERE/runs/${TAG}.jsonl"
python3 "$HERE/zone_hash_check.py" || exit 1
cd "$REPO"
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source python_env/bin/activate
LOG="$HERE/runs/${TAG}.log"
FLAGS="--run-all"
[[ "${S2_DEV:-0}" == "1" ]] && FLAGS="--dev $FLAGS"
[[ "${S2_DEV:-0}" != "1" && "${S2_NOPROF:-0}" != "1" ]] && FLAGS="--profile $FLAGS"
timeout 2400 scripts/run_safe_pytest.sh $FLAGS "$HERE/test_s2.py" --import-mode=prepend -q > "$LOG" 2>&1
grep -E "PASS|FAIL|Error|error:|SAFE_PYTEST|passed|failed|skipped|assert|Timeout|hang|HANG|triage" "$LOG" | \
    grep -v "^PRECOMPILE\|waiting for device" | tail -30
[[ "$FLAGS" != *--profile* ]] && exit 0
CSV=$(ls -t "$TT_METAL_PROFILER_DIR"/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
echo "CSV: $CSV"
[[ -n "$CSV" ]] && python3 "$HERE/report.py" "$CSV" "$HERE/runs/${TAG}.jsonl" --append "$HERE/results.jsonl" --tag "$TAG"
[[ -n "$CSV" ]] && cp "$CSV" "$HERE/runs/${TAG}_ops_perf.csv"
[[ -f "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" ]] && \
    gzip -c "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" > "$HERE/runs/${TAG}_device.csv.gz"
rm -rf "$TT_METAL_PROFILER_DIR"
