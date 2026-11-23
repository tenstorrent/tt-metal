#!/bin/bash
# usage: run.sh <label> "<TAIL defines a;b=1>" ["<WQB_KNOBS>"]   (from the tt-metal root)
set -e
D=ttnn/ttnn/operations/wq_b_linear/perf_experiments/tail_store
source python_env/bin/activate
rm -rf $D/prof/$1
TAIL="$2" WQB_KNOBS="$3" TT_METAL_PROFILER_DIR=$PWD/$D/prof/$1 scripts/run_safe_pytest.sh --profile $D/test_tail_bench.py::test_tail_bench -s 2>&1 | grep -E "^PCC|^TAIL_DEF|passed|failed|error:|Error" | grep -v "bind worker" | head -20
python3 $D/tail.py $D/prof/$1/.logs/profile_log_device.csv | grep -E "spans|median|TRISC_1-K|TRISC_2-K|y_wait|y_issue|y_barrier|BRISC-K|last"
