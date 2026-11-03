#!/bin/bash
# Usage: WDP_VARIANTS=baseline,la1 WDP_CASES=8x512 run.sh  -> "<case> <variant> <ns>" per row (parametrize order).
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=/localdev/kgrujcic/tt-metal-codegen
cd "$REPO" && source python_env/bin/activate
mkdir -p "$HERE/_logs"
tag="$(date +%H%M%S)"
LINEAR_GELU_PERF_PROBES=1 scripts/run_safe_pytest.sh --profile --run-all "$HERE/test_wdp.py" -p no:randomly -s --import-mode=prepend \
    >"$HERE/_logs/$tag.log" 2>&1
grep -E "WDP_RESULT|passed|failed|FAILED|SAFE_PYTEST_RESULT" "$HERE/_logs/$tag.log" | grep -v "pcc=nan"
# Siblings share the profiler output dir: take the CSV this run reported, not the newest one.
csv=$(grep -oE "/[^ ]*ops_perf_results_[0-9_]+\.csv" "$HERE/_logs/$tag.log" | tail -1)
echo "csv $csv"
cp "$(dirname "$csv")/profile_log_device.csv" "$HERE/_logs/$tag.csv" 2>/dev/null
paste -d' ' <(for c in ${WDP_CASES//,/ }; do for v in ${WDP_VARIANTS//,/ }; do echo "$c $v"; done; done) \
    <(awk -F',' 'NR>1 && /GenericOp/ {print $19}' "$csv") | tee -a "$HERE/results.log"
echo "log tag $tag"
