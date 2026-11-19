#!/usr/bin/env bash
# One profiled session of test_tsr.py: TSR_CASES=... TSR_VARIANTS=... [TSR_ORDER=alt] run.sh <tag> [extra pytest args]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"; shift
export TSR_LOG="${TAG}.jsonl"
rm -f "$HERE/runs/${TAG}.jsonl"
cd "$REPO"
source python_env/bin/activate
out=$(timeout 2400 scripts/run_safe_pytest.sh --profile --run-all --no-precompile "$HERE/test_tsr.py" --import-mode=prepend -q "$@" 2>&1)
echo "$out" | grep -E "passed|failed|SAFE_PYTEST_RESULT|HANG|triage|Error|error:" | grep -v "^PRECOMPILE" | tail -8
CSV=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
echo "CSV: $CSV"
[[ -n "$CSV" ]] && python3 "$HERE/report.py" "$CSV" "$HERE/runs/${TAG}.jsonl" --append "$HERE/results.jsonl" --tag "$TAG" && cp "$CSV" "$HERE/runs/${TAG}_ops_perf.csv"
