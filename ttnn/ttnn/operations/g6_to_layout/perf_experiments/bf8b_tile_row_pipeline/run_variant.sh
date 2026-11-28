#!/bin/bash
# run_variant.sh <tag> -- env G6_* as in test_rowpipe_bench.py. One profiled process; copies the
# device CSVs into runs/<tag>/ right after the run (the slot's generated/profiler/.logs is shared).
set -u
ROOT=/localdev/pmilojevic/campaign/bh-lb-18/slots/chip2
HERE=$ROOT/ttnn/ttnn/operations/g6_to_layout/perf_experiments/bf8b_tile_row_pipeline
TAG=$1
cd $ROOT
export TT_METAL_HOME=$ROOT PYTHONPATH=$ROOT TRACY_NO_WEB_SERVER=1
source python_env/bin/activate
OUT=$HERE/runs/$TAG
mkdir -p $OUT
scripts/run_safe_pytest.sh --profile --no-py-zones --no-precompile \
    $HERE/test_rowpipe_bench.py --import-mode=prepend > $OUT/log.txt 2>&1
cp generated/profiler/.logs/cpp_device_perf_report.csv generated/profiler/.logs/profile_log_device.csv $OUT/ 2>/dev/null
grep -E "^RB|RB_OK|FAILED|Error:|bit_mismatch|SAFE_PYTEST_RESULT|Device kernel duration perf summary" $OUT/log.txt | cut -c1-400
python3 - "$OUT/cpp_device_perf_report.csv" <<'PY'
import csv, sys, statistics
rows = list(csv.DictReader(open(sys.argv[1])))
k = [float(r["DEVICE KERNEL DURATION [ns]"]) for r in rows]
print("KERNEL_NS all:", [int(x) for x in k])
if len(k) > 1:
    print("KERNEL_NS median(launch 2..N): %.0f" % statistics.median(k[1:]))
for col in ("DEVICE BRISC KERNEL DURATION [ns]", "DEVICE NCRISC KERNEL DURATION [ns]", "DEVICE TRISC2 KERNEL DURATION [ns]"):
    v = [float(r[col]) for r in rows[1:] if r[col]]
    if v:
        print(col.split()[1], "median %.0f" % statistics.median(v))
PY
