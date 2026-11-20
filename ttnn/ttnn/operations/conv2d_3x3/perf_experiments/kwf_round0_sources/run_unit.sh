#!/bin/bash
# usage: run_unit.sh <label> [--dev] [-k expr]: the kw-fold (+ the new round-0 test) and HEIGHT unit files retargeted
# at op_grad/ (test_krs_unit.py).
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_round0_sources
mkdir -p $E/runs
label=$1; shift
KRS_OP=${KRS_OP:-grad} timeout 3000 scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend "$@" $E/test_krs_unit.py > $E/runs/log_unit_$label.txt 2>&1
grep -v "riscv-tt-elf-g++" $E/runs/log_unit_$label.txt | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|^FAILED|^ERROR|QUEUE" | tail -14
