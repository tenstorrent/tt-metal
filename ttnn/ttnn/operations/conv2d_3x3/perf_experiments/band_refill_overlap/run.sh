#!/bin/bash
# usage: run.sh <label> <cases> <variants> [extra env...]   e.g. run.sh t1 idx41,idx35 base,bro BRO_ITERS=3
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]); records (blocking,
# PCC, sha1, bit-identity vs the first variant) -> runs/<label>.jsonl.
# ACQ_DEADLINE (s, default 300): give up (exit 9, nothing ran) if the shared device lock is not acquired by then, so a
# caller with a bounded wall-clock never kills a run that holds the device.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/band_refill_overlap
label=$1; cases=$2; variants=$3; shift 3
mkdir -p $E/runs
log=$E/runs/$label.log
rm -f $log
env BRO_LOG=$label BRO_CASES=$cases BRO_VARIANTS=$variants "$@" scripts/run_safe_pytest.sh ${DEV:+--dev} --profile --run-all $E/test_bro.py > $log 2>&1 &
pid=$!
start=$(date +%s)
while kill -0 $pid 2>/dev/null; do
    if grep -q "Device lock acquired" $log; then
        wait $pid
        break
    fi
    if (( $(date +%s) - start > ${ACQ_DEADLINE:-300} )); then
        pkill -TERM -P $pid 2>/dev/null
        kill -TERM $pid 2>/dev/null
        wait $pid 2>/dev/null
        echo "LOCK_TIMEOUT after ${ACQ_DEADLINE:-300}s (nothing ran)"
        exit 9
    fi
    sleep 2
done
out=$(cat $log)
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage|Error|error:|TOTAL_RUNTIME" | cut -c1-400 | tail -20
echo "$out" | grep -E "^BRO:" | python3 -c '
import json,sys
for l in sys.stdin:
    r=json.loads(l[4:]); b=r.get("blocking",{})
    print(r["case"], r["variant"], r["status"], "bit" if r.get("bit_identical") else ("DIFF %s ulp %s"%(r.get("frac_diff"),r.get("max_ulp"))), "pcc %.7f"%r.get("pcc",0), "bm",b.get("bm"),"bk",b.get("bk"),"nb",b.get("m_blocks"),"depth",b.get("im2col_depth"),"stg",b.get("bro_nstg"),b.get("bro_chunk_px"), r.get("error",""))'
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
[ -n "$csv" ] && gzip -c "$csv" > $E/runs/${label}_ops.csv.gz
[ -n "$csv" ] && [ -f generated/profiler/.logs/profile_log_device.csv ] && gzip -c generated/profiler/.logs/profile_log_device.csv > $E/runs/${label}_dev.csv.gz
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
