#!/usr/bin/env bash
# One profiled bench session: env QPB_* as test_qpack_bench.py; usage: bench_run.sh <tag>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"
cd "$REPO"
source python_env/bin/activate
export QPB_BENCH=1 QPB_TAG="$TAG"
out=$(timeout 580 scripts/run_safe_pytest.sh --profile --run-all --no-precompile "$HERE/test_qpack_bench.py" --import-mode=prepend -s 2>&1)
echo "$out" > "$HERE/runs/bench_${TAG}.log"
echo "$out" | grep -E "passed|failed|SAFE_PYTEST_RESULT|HANG|triage|Error|error:" | grep -v "^PRECOMPILE" | tail -10
CSV=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[[ -n "$CSV" ]] && python3 "$HERE/bench_report.py" "$CSV" "$TAG"
