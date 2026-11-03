#!/bin/bash
# Usage: BRB_VARIANTS=baseline,first BRB_CASES=8x512,... run.sh -> "<variant> <case> <ns>" per row (appended to results.log).
HERE="$(cd "$(dirname "$0")" && pwd)"
cd /localdev/kgrujcic/tt-metal-codegen && source python_env/bin/activate
variants=${BRB_VARIANTS:-baseline,first}
cases=${BRB_CASES:-8x512}
LINEAR_GELU_PERF_PROBES=1 BRB_VARIANTS=$variants BRB_CASES=$cases scripts/run_safe_pytest.sh --profile --run-all \
    "$HERE/test_brb.py" --import-mode=prepend -p no:randomly -s >/tmp/brb_$$.log 2>&1
grep -E "passed|failed" /tmp/brb_$$.log | tail -1
grep -E "BRB_PCC" /tmp/brb_$$.log | grep -v nan | awk '$5 < 0.9999 {print "PCC FAIL", $0}'
csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
paste -d' ' <(for v in ${variants//,/ }; do for c in ${cases//,/ }; do echo "$v $c"; done; done) \
    <(awk -F',' 'NR>1 && /GenericOp/ {print $19}' "$csv") | tee -a "$HERE/results.log"
