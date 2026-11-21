#!/bin/bash
# usage: wait.sh <label>: follow a run.sh run for up to WAIT_S seconds (default 560); exit 75 while it is not done.
cd /localdev/pmilojevic/workspaces/opgen
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/whole_k_rowstream
label=$1
log=$E/runs/$label.log
waited=0
while [ ! -f $E/runs/$label.done ]; do
    if [ $waited -ge ${WAIT_S:-560} ]; then
        if grep -q "Device lock acquired" $log; then echo "RUNNING: $(grep -c 'RS:' $log) records so far"; else echo "QUEUED"; fi
        exit 75
    fi
    sleep 5
    waited=$((waited + 5))
done
out=$(cat $log)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage" | tail -5
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && [ -f "$csv" ] && gzip -c "$csv" > $E/runs/${label}_ops.csv.gz
[ -n "$csv" ] && [ -f "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
# Keep the run log small: drop the per-op lines.
grep -E "RS:|passed|failed|SAFE_PYTEST|HANG|triage|Error|error|PROFILER" $log > $log.tmp; mv $log.tmp $log
