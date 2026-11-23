#!/bin/bash
# usage: dom.sh label "cells-regex"  -> base then stage, one run each
cd /home/malimpic/localdev/2026_11_23/1317_malimpic_distributed_rms_norm_all_gather_gen/clones/all_gather_rms_norm_run1/tt-metal
source python_env/bin/activate
R=ttnn/ttnn/operations/all_gather_rms_norm/perf_experiments/own_entry_port_l1_staging/results
H=tests/ttnn/unit_tests/operations/all_gather_rms_norm/perf_experiments/own_entry_port_l1_staging/harness/perf_stage.py
for op in ${ORDER:-base stage}; do
  if [ -n "$2" ]; then sel="AGRN_PERF_CELLS=$2"; else sel="CCL_PERF_CASES=${CASE:-0}"; fi
  env AGRN_EXP_OP=$op "$sel" CCL_PERF_OUT=$R/dom_$1_$op timeout 900 scripts/run_safe_pytest.sh --no-precompile $H -s > /tmp/dom_$1_$op.log 2>&1
  echo "$1 $op $(grep -E '^CCL_PERF .*: [0-9]+ experiments' /tmp/dom_$1_$op.log | wc -l) cells: $(grep -A2 '^CCL_PERF ' /tmp/dom_$1_$op.log | grep -E 'fastest|slowest' | awk '{print $3}' | tr '\n' ' ') $(grep SAFE_PYTEST_RESULT /tmp/dom_$1_$op.log)"
done
