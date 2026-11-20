#!/bin/bash
# usage: [SWS_OP=copy SWS_RESIDENT=1 SWS_UNIT_MODULES=..] run_unit.sh <label> [--dev] : unit files retargeted (test_sws_unit.py)
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/s2_weight_slots
label=$1; shift
rm -f $E/runs/domain_$label.log
SWS_LOG=$PWD/$E/runs/domain_$label.log timeout 3000 scripts/run_safe_pytest.sh "$@" --run-all --precompile --import-mode=prepend $E/test_sws_unit.py > $E/runs/log_unit_$label.txt 2>&1
grep -v "riscv-tt-elf-g++" $E/runs/log_unit_$label.txt | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|^FAILED|^ERROR" | tail -12
[ -f $E/runs/domain_$label.log ] && sort $E/runs/domain_$label.log | uniq -c
