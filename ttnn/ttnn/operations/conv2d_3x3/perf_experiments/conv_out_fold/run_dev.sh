#!/bin/bash
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/conv_out_fold
COF_LOG=dev1 COF_CASES=r5,33 COF_VARIANTS=fold timeout 3000 scripts/run_safe_pytest.sh --dev --run-all $E/test_fold.py > $E/runs/dev1.log 2>&1
echo "EXIT $?" >> $E/runs/dev1.log
