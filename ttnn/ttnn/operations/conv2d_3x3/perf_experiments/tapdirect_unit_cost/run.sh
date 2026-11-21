#!/bin/bash
# usage: run.sh <label> <cases> <variants> [extra env...]   e.g. run.sh r1 42,41,42n base,epi
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]);
# PCC / bit-identity / ulp -> runs/<label>.jsonl. Gives up (DEFERRED) if the device stays busy for BE_MAXWAIT s.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/tapdirect_unit_cost
label=$1; cases=$2; variants=$3; shift 3
env BE_LOG=$label BE_CASES=$cases BE_VARIANTS=$variants ${TU_ZONES:+TU_COPY_DEV=/tmp/tu_${label}_dev.csv} "$@" $E/safe_run.sh $E/runs/$label.log --profile --run-all $E/test_tu.py
rc=$?
[ $rc -eq 75 ] && exit 75
grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage" $E/runs/$label.log | cut -c1-300 | tail -10
csv=$(grep "PROFILER CSV:" $E/runs/$label.log | sed 's/.*CSV: //')
[ -n "$csv" ] && cp "$csv" $E/runs/${label}_ops.csv
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | tee $E/runs/$label.txt
python3 - $E/runs/$label.jsonl <<'PY'
import json, sys
try:
    lines = open(sys.argv[1]).readlines()
except FileNotFoundError:
    lines = []
for l in lines:
    r = json.loads(l)
    print(r["case"], r["variant"], r.get("status"), r.get("error", ""), "pcc=%s" % r.get("pcc"), "mae=%s" % r.get("mean_abs_err"),
          "bitid=%s" % r.get("bit_identical_to_first"), "fdiff=%.4f" % (r.get("frac_diff_vs_first") or 0), r.get("blocking", ""))
PY
# TU_ZONES=1: per-stage zone report of the run's device log -> runs/<label>_zones.txt
if [ -n "$TU_ZONES" ]; then
  python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/zone_report.py /tmp/tu_${label}_dev.csv > $E/runs/${label}_zones.txt 2>&1
  if [ -n "$TU_SEQ" ]; then
    python3 $E/zone_seq.py /tmp/tu_${label}_dev.csv --zones $TU_SEQ ${TU_SEQ_ARGS} > $E/runs/${label}_seq.txt 2>&1
  fi
  rm -f /tmp/tu_${label}_dev.csv
fi
