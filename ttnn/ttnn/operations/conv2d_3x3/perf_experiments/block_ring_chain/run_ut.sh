#!/usr/bin/env bash
# A test file of this dir (default test_block_sharded_grad.py) on grad/ (or BRC_UT_PKG), in the foreground.
#   [BRC_DEV=1] [BRC_UT_FILE=...] run_ut.sh <tag> [pytest args]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"; shift
cd "$REPO"
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source python_env/bin/activate
mkdir -p "$HERE/runs"
LOG="$HERE/runs/${TAG}.log"
FLAGS="--run-all"
[[ "${BRC_DEV:-0}" == "1" ]] && FLAGS="--dev $FLAGS"
scripts/run_safe_pytest.sh $FLAGS "$HERE/${BRC_UT_FILE:-test_block_sharded_grad.py}" --import-mode=prepend -q "$@" > "$LOG" 2>&1
grep -E "FAILED|Error|error:|SAFE_PYTEST|passed|failed|Timeout|HANG|triage" "$LOG" | \
    grep -v "^PRECOMPILE\|waiting for device" | tail -30
