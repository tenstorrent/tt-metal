#!/usr/bin/env bash
# bench.sh <case> <variant> [tag] -> median DEVICE KERNEL DURATION of launches 2..12 for the bench op copy
cd /localdev/pmilojevic/campaign/bh-qb-15/slots/chip0
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD; unset TT_METAL_SIMULATOR TT_METAL_PROFILER_DIR
source python_env/bin/activate
c=$1; v=$2; tag=${3:-0}
D=ttnn/ttnn/operations/g6_reshape/perf_experiments/two_mover_split; L=$D/logs; mkdir -p $L
log=$L/${c}_${v}_${tag}.log
for try in 1 2; do
  TMS_VARIANT=$v BENCH_CASES=$c BENCH_IMPL=tms TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile --no-py-zones $D/bench_tms.py::test_bench --import-mode=prepend > $log 2>&1
  csv=$(grep -o 'PROFILER CSV: .*' $log | sed 's/PROFILER CSV: //')
  [ -n "$csv" ] && break
done
grep -E 'BENCH_OK|BENCH_TMS|FAILED|Error' $log | head -4
python3 - "$csv" "$c" "$v" <<'PY'
import sys,csv,statistics
p,c,v=sys.argv[1:]
rows=list(csv.DictReader(open(p)))
d=[int(r['DEVICE KERNEL DURATION [ns]']) for r in rows]
m=d[1:] if len(d)>1 else d
print(f"RESULT {c} {v} n={len(m)} median={statistics.median(m):.0f} min={min(m)} max={max(m)}")
PY
echo CSV $csv
