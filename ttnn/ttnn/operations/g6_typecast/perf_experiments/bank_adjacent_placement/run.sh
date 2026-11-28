#!/bin/bash
# usage: run.sh "CASE|mode;CASE|mode;..."   (one profiler session; 12 launches per item, median of 2..12)
cd /localdev/pmilojevic/campaign/bh-qb-15/slots/chip1; source python_env/bin/activate 2>/dev/null; export TT_METAL_HOME=$PWD
D=ttnn/ttnn/operations/g6_typecast/perf_experiments/bank_adjacent_placement
log=$(mktemp /tmp/plbench.XXXX); ord=$(mktemp /tmp/plorder.XXXX)
PL_ITEMS="$1" PL_ORDER=$ord TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile --no-py-zones $D/test_place.py::test_items --import-mode=prepend > $log 2>&1
rc=$?
grep "PL_CHECK" $log | grep -v "ok=True placed=None" | cut -c1-220
csv=$(grep -o '/[^ ]*ops_perf_results_[^ ]*\.csv' $log | tail -1)
if [ -z "$csv" ] || grep -q " failed\| error" $log; then echo "FAIL rc=$rc log=$log"; grep -E "Error|error:|FAILED|assert" $log | head -20; exit 1; fi
cp $ord /tmp/pl_last_order.txt
python3 $D/parse.py $ord $csv || { echo "log kept: $log"; exit 1; }
rm -f $log
