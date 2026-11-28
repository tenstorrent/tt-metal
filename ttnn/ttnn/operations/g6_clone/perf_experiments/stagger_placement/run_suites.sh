#!/usr/bin/env bash
# Golden + unit suites with the rule installed through stagger_plugin (the op files stay untouched).
# run_suites.sh <logdir> [PLACE value, default rule]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
set +u; source python_env/bin/activate; set -u
export TT_METAL_HOME=$PWD
E=ttnn/ttnn/operations/g6_clone/perf_experiments/stagger_placement
R=generated/meta/g6_clone/lanes/perf1_stagger_placement; mkdir -p $R; cp $E/stagger_plugin.py $R/
export PLACE_ROOT=$PWD/$E PLACE=${2:-rule}
export PYTHONPATH=$PWD/$R${PYTHONPATH:+:$PYTHONPATH}
L=$1; mkdir -p $L
scripts/run_safe_pytest.sh --run-all eval/golden_tests/g6_clone/ -p stagger_plugin -s > $L/golden.log 2>&1; echo "golden rc=$?"
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/g6_clone/ -p stagger_plugin -s > $L/unit.log 2>&1; echo "unit rc=$?"
for f in golden unit; do echo "== $f: $(grep -ac 'STAGGER_PLUGIN installed' $L/$f.log) plugin lines"; grep -a "passed\|failed\|SAFE_PYTEST_RESULT" $L/$f.log | tail -3; done
