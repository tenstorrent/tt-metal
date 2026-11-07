#!/bin/bash
# Usage: run.sh <tag> <pytest -k filter>   (env: SC_VARIANTS, SC_REPEAT, SC_ONLY_FORK)
# Runs the bench under the profiler, snapshots the device log + ops CSV into results/<tag>/ at once
# (generated/profiler is shared with sibling benches), then prints the labelled ops CSV and timeline.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"; shift
OUT="$HERE/results/$TAG"; mkdir -p "$OUT"; rm -f "$OUT"/*
export SC_LOG="$OUT/calls.jsonl"
cd "$REPO";
true --profile --run-all ttnn/ttnn/operations/mhc_post/perf_experiments/startup_coef/test_startup_coef.py -k "$1" --import-mode=prepend 2>&1 | tee "$OUT/pytest.log" | grep -E "passed|failed|error|SAFE_PYTEST|assert" | tail -15
cp generated/profiler/.logs/profile_log_device.csv "$OUT/" 2>/dev/null
CSV=$(grep -o "generated/profiler/reports/[^ ]*ops_perf_results[^ ]*\.csv" "$OUT/pytest.log" | tail -1)
[ -n "$CSV" ] && cp "$CSV" "$OUT/ops.csv"
[ -f "$OUT/ops.csv" ] && python3 "$HERE/../coef_expand/label_csv.py" "$OUT/ops.csv" "$SC_LOG"
python3 "$HERE/timeline.py" "$OUT/profile_log_device.csv" "$SC_LOG"
