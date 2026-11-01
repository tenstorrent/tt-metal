#!/bin/bash
# rep.sh <label> <n> [LN_KNOBS] [-k filter]: n profiled runs; per-row values and medians -> logs/rep_<label>.txt
cd /localdev/kgrujcic/tt-metal-codegen
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/half_band
k=${4:-focus}
out=$here/logs/rep_$1.txt
: > /tmp/ln4/rep_vals.txt
for i in $(seq $2); do
  PYTEST_ADDOPTS="-k '$k'" LN_KNOBS="$3" scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/layer_norm_tile/test_perf_bench.py > /tmp/ln4/rep_run.log 2>&1
  csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
  python3 -c "
import csv,sys
rows=list(csv.DictReader(open('$csv')))
print(' '.join(f\"{int(r['DEVICE KERNEL DURATION [ns]'])/1000:.1f}\" for r in rows))" >> /tmp/ln4/rep_vals.txt
done
python3 -c "
import statistics
vals=[list(map(float,l.split())) for l in open('/tmp/ln4/rep_vals.txt') if l.strip()]
cols=list(zip(*vals))
print('$1 [$3] k=$k')
for l in vals: print('  ', l)
print('  median', [round(statistics.median(c),2) for c in cols])
" | tee -a $out
