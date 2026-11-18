#!/bin/bash
# usage: sweep.sh <label> <test path (+ ::id)> [-k expr ...]
# env: HXW_SWAP=1 / HXW_DEFINES=... (candidate), KNOBS (GENIFY_MATMUL_KNOBS), CHECK=1
# Prints per op call: test id, plan (HXWPLAN) and kernel_us; appends to sweep_<label>.txt
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2_raw_kwalk_hxw
cd "$ROOT"
source python_env/bin/activate
LABEL=$1; shift
export GENIFY_MATMUL_PERF=1 GENIFY_MATMUL_KNOBS="$KNOBS" PYTHONPATH="$D:$PYTHONPATH"
[ -n "$CHECK" ] && export GENIFY_MATMUL_CHECK=1
LOG=$D/last_sweep.log
timeout 5400 scripts/run_safe_pytest.sh --run-all --profile -s -p hxw_plugin "$@" > $LOG 2>&1
echo "exit=$?"
grep -E "FAILED|error:|passed|failed|triage|CHECK .*pcc" $LOG | cut -c1-250 | tail -8
csv=$(grep "PROFILER CSV" $LOG | sed 's/.*CSV: //' | tail -1)
python3 - "$csv" "$LOG" "$LABEL" "$D/sweep_$LABEL.txt" <<'PY'
import csv, sys, re
rows = [r for r in csv.DictReader(open(sys.argv[1]))] if sys.argv[1] else []
ops = [r for r in rows if r["OP CODE"].startswith("GenericOp")]
plans = [l.strip() for l in open(sys.argv[2], errors="replace") if l.startswith("HXWPLAN ")]
out = open(sys.argv[4], "a")
if len(ops) != len(plans):
    print(f"WARNING ops={len(ops)} plans={len(plans)}")
plans = plans[len(plans) - len(ops):] if len(plans) >= len(ops) else plans
for p, r in zip(plans, ops):
    line = f"{sys.argv[3]} {p[8:]} kernel_us={int(r['DEVICE KERNEL DURATION [ns]'])/1000:.1f}"
    print(line); out.write(line + "\n")
PY
