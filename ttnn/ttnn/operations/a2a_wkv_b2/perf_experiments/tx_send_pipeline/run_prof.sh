#!/bin/bash
# usage (from the tt-metal root, login node): run_prof.sh <TXP> <A2A_ABLATE or ""> <tag>
# Runs probe_tx_profile.py under the device profiler on the reserved node, copies the CSV out at once, prints
# trace_gaps + stage zones (+ TXD accumulators for *_d variants) into results/<tag>.txt, deletes the CSV.
set -e
D=ttnn/ttnn/operations/a2a_wkv_b2/perf_experiments/tx_send_pipeline
mkdir -p $D/results
CSV=$D/results/$3.csv
timeout 1200 /data/dstoiljkovic/bin/on-node "TXP=$1 A2A_ABLATE=$2 TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh $D/probe_tx_profile.py -o addopts= --import-mode=prepend -sv > $D/results/$3.log 2>&1; rc=\$?; cp generated/profiler/.logs/profile_log_device.csv $CSV; exit \$rc" || { echo "RUN FAILED"; tail -5 $D/results/$3.log; }
for i in $(seq 1 30); do [ -s $CSV ] && break; sleep 2; done
{
  echo "TXP=$1 A2A_ABLATE=$2"
  grep -E "SAFE_PYTEST_RESULT|PASSED|FAILED" $D/results/$3.log | head -3
  python3 ttnn/ttnn/operations/a2a_wkv_b2/perf/trace_gaps.py $CSV
  python3 ttnn/ttnn/operations/a2a_wkv_b2/perf/stage_zones.py $CSV 2>&1 | cat
  python3 $D/txd.py $CSV | grep MEAN
} > $D/results/$3.txt
gzip -f $CSV
cat $D/results/$3.txt
