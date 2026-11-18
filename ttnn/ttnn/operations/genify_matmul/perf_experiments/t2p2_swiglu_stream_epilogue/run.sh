#!/bin/bash
# usage: run.sh <label> <ids|""> "<knobs>" [test file (+::id / -k ...)]
#   env: SS_SWAP=1 SS_DEFINES=GM_SS_STREAM (candidate), SS_COMPARE=1 (base vs variant equality), CHECK=1 (PCC),
#        ZONES=1 (zone summary), FORCE / BLOCK (perf-dir split overrides)
# Prints SSPLAN / SSEQ / CHECK lines and per-GenericOp kernel_us; appends to runs.log.
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2p2_swiglu_stream_epilogue
cd "$ROOT"
source python_env/bin/activate
LABEL=$1; IDS=$2; KN=$3; shift 3
FILE=${1:-tests/ttnn/unit_tests/operations/genify_matmul/perf/test_genify_matmul_perf_r15.py}; shift
export GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_PLAN=1 GENIFY_MATMUL_ONLY="$IDS" GENIFY_MATMUL_KNOBS="$KN"
export GENIFY_MATMUL_FORCE="$FORCE" GENIFY_MATMUL_BLOCK="$BLOCK" PYTHONPATH="$D:$PYTHONPATH"
[ -n "$CHECK" ] && export GENIFY_MATMUL_CHECK=1
LOG=$D/last_run_$LABEL.log
timeout 3000 scripts/run_safe_pytest.sh --run-all --profile -s -p ss_plugin $FILE "$@" > $LOG 2>&1
echo "exit=$?"
csv=$(grep "PROFILER CSV" $LOG | sed 's/.*CSV: //' | tail -1)
grep -E "^PLAN|SSPLAN|SSEQ|CHECK |FAILED|error:|passed|failed|triage" $LOG | sed "s/epilogue=.*//" | cut -c1-300 | head -60
python3 - "$csv" "$LOG" "$LABEL" "$D/runs.log" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]))] if sys.argv[1] else []
ops = [r for r in rows if r["OP CODE"].startswith("GenericOp")]
ids = []
for l in open(sys.argv[2], errors="replace"):
    if l.startswith("SSPLAN "):
        ids.append(l.split()[1])
out = open(sys.argv[4], "a")
for i, r in enumerate(ops):
    lab = ids[i] if i < len(ids) else str(i)
    line = f"{sys.argv[3]} {lab} cores={r['CORE COUNT']} kernel_us={int(r['DEVICE KERNEL DURATION [ns]'])/1000:.1f}"
    print(line); out.write(line + "\n")
PY
[ -n "$ZONES" ] && python ttnn/ttnn/operations/genify_matmul/agent_logs/t2p1_zone_summary.py $(dirname $csv)/profile_log_device.csv
[ -n "$SAVE" ] && cp $(dirname $csv)/profile_log_device.csv $D/prof_$LABEL.csv
