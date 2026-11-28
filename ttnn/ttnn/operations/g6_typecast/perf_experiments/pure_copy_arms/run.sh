#!/bin/bash
# usage: run.sh "case|variant;case|variant;..."
cd /localdev/pmilojevic/campaign/bh-qb-15/slots/chip1; source python_env/bin/activate 2>/dev/null; export TT_METAL_HOME=$PWD
D=ttnn/ttnn/operations/g6_typecast/perf_experiments/pure_copy_arms
log=$(mktemp /tmp/pcbench.XXXX); ord=$(mktemp /tmp/pcorder.XXXX)
PC_ITEMS="$1" PC_ORDER=$ord TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile --no-py-zones $D/test_pure_copy.py::test_items --import-mode=prepend > $log 2>&1
rc=$?
grep "PC_CHECK" $log | grep -v "exact=True"
csv=$(grep -o '/[^ ]*ops_perf_results_[^ ]*\.csv' $log | tail -1)
if [ -z "$csv" ] || [ $rc -ne 0 ]; then echo "FAIL rc=$rc log=$log"; grep -E "Error|error:|FAILED|assert" $log | head -20; exit 1; fi
echo "csv=$csv"; python3 $D/parse.py $ord $csv || { echo "log kept: $log"; exit 1; }
echo "checks: $(grep -c 'exact=True' $log) exact"; rm -f $log
