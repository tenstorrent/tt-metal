#!/bin/bash
# Usage: run_perf.sh <test>[<variant>-<case>] ...   e.g. run_perf.sh "test_perf[mac-full-conv1d-004]"
#   test_perf[<body>-<full|nodm>-<case>]            body: lib | q | raw | mac | op (unmodified op)
#   test_pass[<body>-<full|nodm>-rep<k>-<case>]     GQKV_SILU_REPEAT = k (per-pass SFPU cost = T(k+1) - T(k))
# One op call per test, ONE session, in the order given. No --profile: the device profiler is enabled through the
# environment and read in-process (ttnn.ReadDeviceProfiler), printed as "SCHED_NS <id> <device kernel ns>".
# Profiler artifacts go to a private dir (siblings share generated/profiler); the device log (stage zones, one
# run host ID per op) is kept as $TT_METAL_PROFILER_DIR/saved/<n>.csv.
ROOT=/localdev/nbabin/2026_11_26/1733_nbabin_gen_qkv_causal_conv1d_silu/clones/gen_qkv_causal_conv1d_silu_run1/tt-metal
F=ttnn/ttnn/operations/gen_qkv_causal_conv1d_silu/perf_experiments/silu_sfpu_schedule/test_silu_sched.py
export TT_METAL_PROFILER_DIR=/tmp/silu_sched_prof
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
mkdir -p $TT_METAL_PROFILER_DIR/saved
rm -f $TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv
cd "$ROOT"
source python_env/bin/activate
ids=()
for v in "$@"; do ids+=("$F::$v"); done
LOG=/tmp/silu_sched_perf_last.log
GQKV_SCHED_PERF=1 scripts/run_safe_pytest.sh --no-jit-server --run-all "${ids[@]}" > $LOG 2>&1
grep -E "SAFE_PYTEST_RESULT|[0-9]+ (passed|failed)" $LOG | tail -2
n=$(ls $TT_METAL_PROFILER_DIR/saved/*.csv 2>/dev/null | wc -l)
saved=$TT_METAL_PROFILER_DIR/saved/$n.csv
cp $TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv $saved 2>/dev/null && echo "DEVICE_LOG $saved"
echo "$n $*" >> $TT_METAL_PROFILER_DIR/saved/index.txt
grep -o "SCHED_NS .*" $LOG | awk 'NF >= 3 && $3 != "None" && !seen[$2]++ {printf "%-40s %10s ns\n", $2, $3}'
