#!/usr/bin/env bash
# STEP A clock-skew probe: run_clock.sh <logdir> [pytest -k expr]. Runs a copy from the git-ignored lane dir
# (a test file under ttnn/ re-imports ttnn under pytest).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
set +u; source python_env/bin/activate; set -u
export TT_METAL_HOME=$PWD
E=ttnn/ttnn/operations/g6_clone/perf_experiments/stagger_placement
R=generated/meta/g6_clone/lanes/perf1_stagger_placement; mkdir -p $R
cp $E/test_clock_skew.py $R/
export SKEW_ROOT=$PWD/$E
L=$1; shift; mkdir -p $L
export SKEW_OUT=$PWD/$L
K=${1:-}
scripts/run_safe_pytest.sh --run-all $R/test_clock_skew.py ${K:+-k "$K"} -s > $L/clock.log 2>&1
echo "rc=$?"
grep -a '^SKEW_\|PASS\|FAIL\|Error' $L/clock.log | head -200
