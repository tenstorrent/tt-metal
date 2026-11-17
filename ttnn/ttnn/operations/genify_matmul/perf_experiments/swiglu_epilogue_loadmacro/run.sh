#!/bin/bash
# usage: run.sh <set> [--profile]   (env: SWG_VARIANTS=a,b for set "probe"/"stress", SWG_CAND for "domain")
cd "$(dirname "$0")/../../../../../.." || exit 1
source python_env/bin/activate
E=ttnn/ttnn/operations/genify_matmul/perf_experiments/swiglu_epilogue_loadmacro
SET=$1; shift
rm -f $E/results_$SET.jsonl
LOG=$E/last_run_$SET.log
SWG_SET=$SET timeout 1500 scripts/run_safe_pytest.sh --run-all "$@" tests/ttnn/unit_tests/operations/genify_matmul/perf_experiments/swiglu_epilogue_loadmacro/test_swiglu_lm_bench.py > $LOG 2>&1
echo "exit=$?"
grep -oE "[a-z_]+\.(hpp|cpp):[0-9]+:[0-9]+: error: .*" $LOG | sort -u | head -20
grep -E "passed|failed" $LOG | tail -3
if [[ " $* " == *" --profile "* ]]; then
  csv=$(grep -oE "PROFILER CSV: .*" $LOG | sed 's/PROFILER CSV: //' | tail -1)
  python3 $E/parse.py "$csv" $E/results_$SET.jsonl
else
  python3 - "$E/results_$SET.jsonl" <<'PY'
import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    print(r['idx'], r['variant'], {k:(round(v,6) if isinstance(v,float) else v) for k,v in r.items() if k in ('out_dtype','fp32_dest','natural_pairs','pcc','rel_rms','max_ulp','mean_ulp','frac_gt_0p5ulp','frac_gt_1ulp','tiny_max_abs','max_ulp32','mean_ulp32')})
PY
fi
