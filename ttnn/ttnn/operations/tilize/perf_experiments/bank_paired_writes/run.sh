#!/bin/bash
# usage: BPW_VARS=a,b BPW_SHAPES=s1,s2 [BPW_MEM=dram|l1|hs] run.sh [--noprof]
# Profiled run (one fresh run per variant x shape); prints ns per (shape, variant) in collection order.
ROOT=/localdev/adamjanovic/2026_09_23/1616_agent_eval_dm/clones/tilize_run1/tt-metal
cd $ROOT; source python_env/bin/activate
T=tests/ttnn/unit_tests/operations/tilize/test_tilize_perf1_bank_paired_writes.py
LOG=$(mktemp /tmp/bpw_XXXX.log)
if [[ "$1" == "--noprof" ]]; then
  timeout 1500 scripts/run_safe_pytest.sh --run-all $T > $LOG 2>&1
  grep -E "passed|failed|SAFE_PYTEST_RESULT|^FAILED|^ERROR" $LOG | tail -8; exit
fi
timeout 1500 scripts/run_safe_pytest.sh --run-all --profile $T > $LOG 2>&1
grep -E "passed|failed|^ERROR|^FAILED" $LOG | tail -4
CSV=$(grep "PROFILER CSV:" $LOG | sed 's/.*PROFILER CSV: //')
echo "log=$LOG csv=$CSV"
python3 - "$CSV" <<'PY'
import csv, os, sys
rows = list(csv.DictReader(open(sys.argv[1])))
ids = []
for s in os.environ.get("BPW_SHAPES", "1x1x16384x64").split(","):
    for v in os.environ.get("BPW_VARS", "F_base").split(","):
        ids.append((s, v))
print(f"n_rows={len(rows)} n_tests={len(ids)}")
for (s, v), r in zip(ids, rows):
    print(f"{s:16s} {v:14s} {r['DEVICE KERNEL DURATION [ns]']:>8s} cores={r['CORE COUNT']}")
PY
