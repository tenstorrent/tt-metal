#!/bin/bash
# usage: run.sh <label> <cases> <variants> [extra env...]   e.g. run.sh r1 42,41,42n base,epi
# Device time per signposted call -> runs/<label>.txt (cmp_windows.py: case|variant  median  [calls]);
# PCC / bit-identity / ulp -> runs/<label>.jsonl. Gives up (DEFERRED) if the device stays busy for BE_MAXWAIT s.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/bias_epilogue
label=$1; cases=$2; variants=$3; shift 3
env BE_LOG=$label BE_CASES=$cases BE_VARIANTS=$variants "$@" $E/safe_run.sh $E/runs/$label.log --profile --run-all $E/test_bias_epi.py
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
