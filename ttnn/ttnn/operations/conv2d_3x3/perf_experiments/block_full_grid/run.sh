#!/usr/bin/env bash
# One profiled session of test_bfg.py into this dir's own profiler folder (siblings share the default one).
#   BFG_CASES=... BFG_VARIANTS=... [BFG_ORDER=alt] run.sh <tag>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"
export BFG_LOG="${TAG}.jsonl"
export TT_METAL_PROFILER_DIR="$HERE/prof/$TAG"
rm -rf "$TT_METAL_PROFILER_DIR" "$HERE/runs/${TAG}.jsonl"
cd "$REPO"
scripts/run_safe_pytest.sh --profile --run-all "$HERE/test_bfg.py" --import-mode=prepend -q 2>&1 | \
    grep -E "PASS|FAIL|Error|error:|SAFE_PYTEST|passed|failed|skipped|assert" | grep -v "^PRECOMPILE" | tail -25
CSV=$(ls -t "$TT_METAL_PROFILER_DIR"/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
echo "CSV: $CSV"
[[ -n "$CSV" ]] && python3 "$HERE/report.py" "$CSV" "$HERE/runs/${TAG}.jsonl" --append "$HERE/results.jsonl" --tag "$TAG"
# Keep the per-op CSV and the gzipped device log; the tracy host logs are hundreds of MB per session.
[[ -n "$CSV" ]] && cp "$CSV" "$HERE/runs/${TAG}_ops_perf.csv"
[[ -f "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" ]] && \
    gzip -c "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" > "$HERE/runs/${TAG}_device.csv.gz"
rm -rf "$TT_METAL_PROFILER_DIR"
