#!/usr/bin/env bash
# usage: TP_VARIANT=base|cand run_tp.sh <outdir> <tag> <case>...   (one profiled process per case)
D=/localdev/pmilojevic/campaign/bh-qb-15/slots/chip2/ttnn/ttnn/operations/g6_copy/perf_experiments/tiny_placement
cd /localdev/pmilojevic/campaign/bh-qb-15/slots/chip2
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
unset TT_METAL_SIMULATOR TT_METAL_PROFILER_DIR
O=$1; TAG=$2; shift 2; mkdir -p $O
for c in "$@"; do
  for attempt in 1 2 3; do
    BENCH_IMPL=${BENCH_IMPL:-new} BENCH_CASES=$c TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile --no-py-zones $D/bench_tp.py::test_bench --import-mode=prepend > $O/${TAG}_$c.log 2>&1
    grep -q "PROFILER CSV" $O/${TAG}_$c.log && break
  done
  grep -h "^BENCH_TP\|^BENCH_OK" $O/${TAG}_$c.log | head -2 | tr '\n' ' '; echo
  python3 generated/meta/g6_copy/r01_repro/med.py $O/${TAG}_$c.log
done
