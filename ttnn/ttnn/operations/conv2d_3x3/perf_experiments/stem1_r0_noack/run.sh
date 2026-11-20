#!/usr/bin/env bash
# One profiled session of test_r0.py into this dir's own profiler folder (siblings share the default one).
#   R0_CASES=... R0_VARIANTS=... [R0_ORDER=alt] run.sh <tag>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"
export R0_LOG="${TAG}.jsonl"
export TT_METAL_PROFILER_DIR="$HERE/prof/$TAG"
rm -rf "$TT_METAL_PROFILER_DIR" "$HERE/runs/${TAG}.jsonl"
cd "$REPO"
scripts/run_safe_pytest.sh --profile --run-all "$HERE/${R0_TEST:-test_r0.py}" --import-mode=prepend -q 2>&1 | \
    grep -E "PASS|FAIL|Error|error:|SAFE_PYTEST|passed|failed|skipped|assert|Timeout|hang" | grep -v "^PRECOMPILE" | tail -30
CSV=$(ls -t "$TT_METAL_PROFILER_DIR"/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
echo "CSV: $CSV"
[[ -n "$CSV" ]] && python3 "$HERE/report.py" "$CSV" "$HERE/runs/${TAG}.jsonl" --append "$HERE/results.jsonl" --tag "$TAG"
[[ -n "$CSV" ]] && cp "$CSV" "$HERE/runs/${TAG}_ops_perf.csv"
[[ -f "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" ]] && \
    gzip -c "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" > "$HERE/runs/${TAG}_device.csv.gz"
mkdir -p /tmp/r0_zone_logs && cp "$TT_METAL_PROFILER_DIR"/.logs/*zone_src_locations.log /tmp/r0_zone_logs/ 2>/dev/null
rm -rf "$TT_METAL_PROFILER_DIR"
