#!/usr/bin/env bash
# One profiled bench session: env S2T_* as test_bench.py; usage: bench_run.sh <tag>
# Joins the profiler CSV's GenericOp rows (2 per variant: correctness run, timed run) with the variant order.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"
cd "$REPO"
source python_env/bin/activate
export S2T_BENCH=1 S2T_TAG="$TAG"
out=$(timeout 1500 scripts/run_safe_pytest.sh ${S2T_DEV:+--dev} --profile --run-all --no-precompile "$HERE/test_bench.py" --import-mode=prepend -s 2>&1)
echo "$out" > "$HERE/runs/bench_${TAG}.log"
echo "$out" | grep -E "s2 taps bench|bit-exact|DIFFERS|DBG|passed|failed|SAFE_PYTEST_RESULT|HANG|triage|Error|error:" | grep -v "^PRECOMPILE" | tail -40
CSV=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
echo "CSV: $CSV"
[[ -n "$CSV" ]] && python3 "$HERE/bench_report.py" "$CSV" "$TAG"
