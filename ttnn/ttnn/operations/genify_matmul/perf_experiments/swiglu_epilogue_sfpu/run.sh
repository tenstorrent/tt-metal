#!/bin/bash
# usage: run.sh <set> [--profile]   -> runs the bench set, prints SWG_RESULT lines, compile errors, and (profile) per-case ns
cd "$(dirname "$0")/../../../../../.." || exit 1
source python_env/bin/activate
E=ttnn/ttnn/operations/genify_matmul/perf_experiments/swiglu_epilogue_sfpu
SET=$1; shift
rm -f $E/results_$SET.jsonl
LOG=$E/last_run_$SET.log
SWG_SET=$SET timeout 1500 scripts/run_safe_pytest.sh --run-all "$@" tests/ttnn/unit_tests/operations/genify_matmul/perf_experiments/swiglu_epilogue_sfpu/test_swiglu_epilogue_bench.py > $LOG 2>&1
echo "exit=$?"
grep -E "error:" $LOG | sed 's|.*swiglu_epilogue_sfpu/||' | sort -u | head -20
grep -E "passed|failed" $LOG | tail -3
grep -E "PROFILER CSV" $LOG
if [[ " $* " == *" --profile "* ]]; then
  csv=$(grep -oE "PROFILER CSV: .*" $LOG | sed 's/PROFILER CSV: //' | tail -1)
  python3 $E/parse.py "$csv" $E/results_$SET.jsonl
else
  python3 - "$E/results_$SET.jsonl" <<'PY'
import json,sys
d={}
for l in open(sys.argv[1]):
    r=json.loads(l); d[r['idx']]=r
for r in [d[k] for k in sorted(d)]:
    print(r['idx'], r['variant'], {k:(round(v,6) if isinstance(v,float) else v) for k,v in r.items() if k in ('out_dtype','fp32_dest','natural_pairs','block_size','unpack_to_dest','pcc','rel_rms','max_ulp','mean_ulp','frac_gt_0p5ulp','frac_gt_1ulp','argmax_ref','argmax_got')})
PY
fi
