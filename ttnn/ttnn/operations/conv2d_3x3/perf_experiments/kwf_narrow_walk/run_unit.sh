#!/bin/bash
# usage: run_unit.sh <label> [--dev]: conv2d_3x3's kw-fold and HEIGHT unit files retargeted at op_grad/ (test_knw_unit.py).
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_narrow_walk
label=$1; shift
KNW_OP=${KNW_OP:-grad} KNW_UNIT_MODULES=${KNW_UNIT_MODULES:-test_conv2d_3x3_kw_fold,test_conv2d_3x3_height_sharded} \
    timeout 3000 scripts/run_safe_pytest.sh "$@" --run-all --precompile --import-mode=prepend $E/test_knw_unit.py > $E/runs/log_unit_$label.txt 2>&1
grep -v "riscv-tt-elf-g++" $E/runs/log_unit_$label.txt | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|^FAILED|^ERROR" | tail -12
