#!/bin/bash
# Usage: GUARD_MODES=legacy,new [GUARD_CASES=...] guard.sh  -> "<mode> <case> <ns>" per row.
set -e
cd "$(dirname "$0")/../../../../../.."
modes=${GUARD_MODES:-legacy,new}
cases=${GUARD_CASES:-8x512,16x256,32x128,8x384,4x512,8x256,2x512,4x256,8x512_dramx,8x512_dramout,8x500,1x32}
LINEAR_GELU_PERF_PROBES=1 GUARD_MODES=$modes GUARD_CASES=$cases scripts/run_safe_pytest.sh --profile \
    tests/ttnn/unit_tests/operations/linear_gelu/probes/test_p1_guard.py -p no:randomly >/tmp/guard_$$.log 2>&1 || true
grep -E "passed|failed" /tmp/guard_$$.log | tail -1
csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
paste -d' ' <(for m in ${modes//,/ }; do for c in ${cases//,/ }; do echo "$m $c"; done; done) \
    <(awk -F',' 'NR>1 && /GenericOp/ {print $19}' "$csv")
