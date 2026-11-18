#!/bin/bash
# run_variant.sh <label> <unused> <interleave chunks> <late_block 0|1> [focus|tall|zones] [round_inc 0|1]
# Runs lti_harness on the op copy through tt-probe (device lock, reset); prints the harness summary only.
set -o pipefail
ROOT=/home/malimpic/localdev/2026_11_17/1458_malimpic_distributed_rms_norm_all_gather_gen/clones/all_gather_rms_norm_kimi_run1/tt-metal
cd $ROOT && source python_env/bin/activate
R=ttnn/ttnn/operations/all_gather_rms_norm_kimi/perf_experiments/late_tail_interleave/results
mode=${5:-focus}
extra=""
[ "$mode" = tall ] && extra="LTI_SHAPE=tall"
[ "$mode" = zones ] && extra="LTI_ZONES=1 TT_METAL_DEVICE_PROFILER=1"
LOG=$R/$1.log
env $extra LTI_IMPL=${LTI_IMPL:-copy} LTI_ROUND_INC=${6:-0} LTI_INTERLEAVE=$3 LTI_LATE_BLOCK=$4 LTI_OUT=$R LTI_LABEL=$1 scripts/tt-probe.sh all_gather_rms_norm_kimi \
  <<< "from ttnn.operations.all_gather_rms_norm_kimi.perf_experiments.late_tail_interleave import lti_harness as h; h.run()" > $LOG 2>&1
rc=$?
grep -v "riscv-tt-elf\|^ccache" $LOG | grep -E "LTI|fastest chip|slowest chip|spread|per chip|DISPATCH|programs per chip|Error|error:|Traceback|assert|TT_PROBE: (Hang|triage)" | cut -c1-300
if [ "$mode" = zones ]; then
  python3 $ROOT/ttnn/ttnn/operations/all_gather_rms_norm_kimi/perf/p1_tail_analyze.py.txt $ROOT/generated/profiler/.logs/profile_log_device.csv
  cp $ROOT/generated/profiler/.logs/profile_log_device.csv $R/$1_profile.csv
fi
echo "rc=$rc"
