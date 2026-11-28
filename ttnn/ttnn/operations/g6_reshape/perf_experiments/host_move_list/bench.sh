#!/usr/bin/env bash
# bench.sh <variant> <case>  -> median DEVICE KERNEL DURATION [ns] of launches 2..N (GenericOp rows), one process.
R=/localdev/pmilojevic/campaign/bh-qb-15/slots/chip0
cd $R
export TT_METAL_HOME=$R PYTHONPATH=$R; unset TT_METAL_SIMULATOR TT_METAL_PROFILER_DIR
source python_env/bin/activate
D=ttnn/ttnn/operations/g6_reshape/perf_experiments/host_move_list
v=$1; c=$2; L=$D/logs; mkdir -p $L
log=$L/${v}_${c}.log
for try in 1 2; do
  TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile --no-py-zones $D/test_hml.py::test_bench -k "$v-$c" \
      --rootdir=$D --confcutdir=$R > $log 2>&1
  csv=$(grep -o 'PROFILER CSV: .*' $log | sed 's/PROFILER CSV: //')
  [ -n "$csv" ] && break
done
grep -E "HML_OK|HML_PLAN|FAILED" $log | head -3
python3 - "$csv" "$v" "$c" <<'PY'
import sys, csv, statistics
p, v, c = sys.argv[1:]
rows = [r for r in csv.DictReader(open(p)) if "GenericOp" in r.get("OP CODE", "GenericOp")]
d = [int(r["DEVICE KERNEL DURATION [ns]"]) for r in rows][1:]
f2l = [int(r["DEVICE KERNEL FIRST TO LAST START [ns]"]) for r in rows][1:]
print(f"RESULT {v} {c} n={len(d)} median={statistics.median(d):.0f} min={min(d)} max={max(d)} f2l_med={statistics.median(f2l):.0f}")
PY
echo "$csv" > $L/${v}_${c}.csvpath
