#!/usr/bin/env bash
# One profiled bench session: env TSR_* as test_bench.py; usage: bench_run.sh <tag>
# Joins the profiler CSV's GenericOp rows (2 per variant: correctness run, timed run) with the variant order.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"
cd "$REPO"
source python_env/bin/activate
export TSR_BENCH=1 TSR_PROF=0 TSR_TAG="$TAG"
out=$(timeout 1500 scripts/run_safe_pytest.sh --profile --run-all --no-precompile "$HERE/test_bench.py" --import-mode=prepend -s 2>&1)
echo "$out" | grep -E "tap shift bench|ns/tap|passed|failed|SAFE_PYTEST_RESULT|HANG|triage|assert" | grep -v "^PRECOMPILE" | tail -30
CSV=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
echo "CSV: $CSV"
[[ -n "$CSV" ]] && python3 "$HERE/bench_report.py" "$CSV" "$TAG"
