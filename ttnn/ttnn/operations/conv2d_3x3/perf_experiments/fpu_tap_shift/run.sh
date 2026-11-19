#!/usr/bin/env bash
# One profiled session of test_fts.py into this dir's own profiler folder (siblings share the default one).
#   FTS_CASES=... FTS_VARIANTS=... [FTS_ORDER=alt] [CONV2D_3X3_PERF_ABLATION=1 CONV2D_3X3_ABLATE=...] run.sh <tag>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"
export FTS_LOG="${TAG}.jsonl"
export TT_METAL_PROFILER_DIR="$HERE/prof/$TAG"
rm -rf "$TT_METAL_PROFILER_DIR" "$HERE/runs/${TAG}.jsonl"
cd "$REPO"
source python_env/bin/activate
scripts/run_safe_pytest.sh --profile --run-all "$HERE/test_fts.py" --import-mode=prepend -q 2>&1 | \
    grep -E "PASS|FAIL|Error|error:|SAFE_PYTEST|passed|failed|skipped|assert|HANG|triage" | grep -v "^PRECOMPILE" | tail -25
CSV=$(ls -t "$TT_METAL_PROFILER_DIR"/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
echo "CSV: $CSV"
[[ -n "$CSV" ]] && python3 "$HERE/report.py" "$CSV" "$HERE/runs/${TAG}.jsonl" --append "$HERE/results.jsonl" --tag "$TAG"
[[ -n "$CSV" ]] && cp "$CSV" "$HERE/runs/${TAG}_ops_perf.csv"
[[ -f "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" ]] && \
    gzip -c "$TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv" > "$HERE/runs/${TAG}_device.csv.gz"
rm -rf "$TT_METAL_PROFILER_DIR"
