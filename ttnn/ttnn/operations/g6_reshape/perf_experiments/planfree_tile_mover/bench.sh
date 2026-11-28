#!/usr/bin/env bash
# bench.sh <case> <variant> [zones]  -> median DEVICE KERNEL DURATION of launches 2..N (+ zone medians)
cd /localdev/pmilojevic/campaign/bh-qb-15/slots/chip0
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD; unset TT_METAL_SIMULATOR TT_METAL_PROFILER_DIR
source python_env/bin/activate
c=$1; v=$2
D=ttnn/ttnn/operations/g6_reshape/perf_experiments/planfree_tile_mover; L=$D/logs; mkdir -p $L
log=$L/${c}_${v}.log
for try in 1 2; do
  G6PF_VARIANT=$v BENCH_CASES=$c TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile --no-py-zones $D/bench_cases.py::test_bench > $log 2>&1
  csv=$(grep -o 'PROFILER CSV: .*' $log | sed 's/PROFILER CSV: //')
  [ -n "$csv" ] && break
done
grep -E 'BENCH_OK|BENCH_VARIANT|FAILED|Error|SAFE_PYTEST_RESULT' $log | head -5
python3 - "$csv" "$c" "$v" <<'PY'
import sys,csv,statistics
p,c,v=sys.argv[1:]
rows=list(csv.DictReader(open(p)))
d=[int(r['DEVICE KERNEL DURATION [ns]']) for r in rows]
m=d[1:] if len(d)>1 else d
print(f"RESULT {c} {v} n={len(m)} median={statistics.median(m):.0f} min={min(m)} max={max(m)} f2l={statistics.median(int(r['DEVICE KERNEL FIRST TO LAST START [ns]']) for r in rows[1:] or rows):.0f}")
PY
python3 $D/kdur.py generated/profiler/.logs/profile_log_device.csv
if [ -n "$3" ]; then python3 $D/zones.py generated/profiler/.logs/profile_log_device.csv; fi
