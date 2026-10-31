#!/bin/bash
# usage: run.sh <label> [cases] -- env assignments...
cd /home/malimpic/localdev/2026_10_30/1730_malimpic_nuke-ccl-ops/clones/distributed_rms_norm_run1/tt-metal
source python_env/bin/activate
label=$1; cases=$2; shift 2
out=/tmp/esw/res/$label
rm -rf $out; mkdir -p $out
env "$@" CCL_PERF_CASES=$cases CCL_PERF_EXPERIMENTS=${EXP:-5} CCL_PERF_OUT=$out timeout 3000 scripts/run_safe_pytest.sh --no-precompile ttnn/ttnn/operations/distributed_rms_norm/perf_experiments/early_small_wave/perf_measure_esw.py -s > $out/log.txt 2>&1
grep -E "SAFE_PYTEST_RESULT|AssertionError|elements differ" $out/log.txt | tail -3
for f in $out/*.json; do python -m eval.ccl_perf show "$f" 2>/dev/null | grep -iE "fastest chip|slowest chip" | head -2 | awk "{print \$1,\$3}" | tr '\n' ' '; echo " <- $(basename "$f" | cut -c1-14)"; done
