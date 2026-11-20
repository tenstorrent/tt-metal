#!/bin/bash
# usage: run.sh <label> <cases> <variants> [--dev] [env...]   e.g. run.sh t1 41,35 base,ring
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]).
# On SIGTERM (a caller's time cap) while still queued for the device lock, the queued run is killed; once it holds the
# device it is left to finish (RMB_DEADLINE makes it skip the variants it has no time for).
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/rm_ring_bank_run
label=$1; cases=$2; variants=$3; shift 3
dev=""; prof="--profile"
if [ "$1" == "--dev" ]; then dev="--dev"; prof=""; shift; fi
log=$E/runs/$label.log
setsid env RMB_LOG=$label RMB_CASES=$cases RMB_VARIANTS=$variants "$@" timeout 3000 scripts/run_safe_pytest.sh $dev $prof --run-all $E/test_rmb.py > $log 2>&1 &
pid=$!
on_term() {
  if ! grep -q "Device lock acquired" $log; then
    kill -- -$pid 2>/dev/null
    echo "RUN.SH: killed while queued for the device"
  else
    echo "RUN.SH: caller timed out while the run holds the device; left running (log $log)"
  fi
  exit 4
}
trap on_term TERM INT
wait $pid
grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|RMB:|error" $log | cut -c1-600 | tail -40
csv=$(grep "PROFILER CSV:" $log | sed 's/.*CSV: //')
[ -n "$csv" ] && gzip -c "$csv" > $E/runs/${label}_ops.csv.gz
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
