#!/bin/bash
# usage: bench.sh "<variants>"   env: HXWB_BLOCK, HXWB_FID, HXWB_FP32, HXWB_BDTYPE, HXWB_GRID, HXWB_CHECK=1
ROOT=/localdev/avelickov/2026_11_17/0142_avelickov_op-genify-matmul/clones/genify_matmul_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/genify_matmul/perf_experiments/t2_raw_kwalk_hxw
cd "$ROOT"
source python_env/bin/activate
export HXWB_VARIANTS="$1"
LOG=$D/last_bench.log
if [ "$HXWB_CHECK" = "1" ]; then
  timeout 3000 scripts/run_safe_pytest.sh --run-all -s tests/ttnn/unit_tests/operations/genify_matmul/perf_experiments/t2_raw_kwalk_hxw/test_hxw_op.py::test_hxw_bench > $LOG 2>&1
else
  timeout 3000 scripts/run_safe_pytest.sh --run-all --profile -s tests/ttnn/unit_tests/operations/genify_matmul/perf_experiments/t2_raw_kwalk_hxw/test_hxw_op.py::test_hxw_bench > $LOG 2>&1
fi
echo "exit=$?"
grep -E "HXWB|FAILED|error:|passed|failed|triage" $LOG | cut -c1-300 | head -40
echo "$(date +%T) bench variants=$1 block=$HXWB_BLOCK fid=$HXWB_FID fp32=$HXWB_FP32 B=$HXWB_BDTYPE check=$HXWB_CHECK" >> $D/runs.log
if [ "$HXWB_CHECK" != "1" ]; then
csv=$(grep "PROFILER CSV" $LOG | sed 's/.*CSV: //' | tail -1)
python3 - "$csv" "$1" "${HXWB_BLOCK:-16,5,4,1,9,160}" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["OP CODE"].startswith("GenericOp")]
vs = sys.argv[2].split(",")
bm, bn, sbh, sbw, bk, k = (int(x) for x in sys.argv[3].split(","))
for v, r in zip(vs, rows[-len(vs):]):
    ns = int(r["DEVICE KERNEL DURATION [ns]"])
    print(f"HXWB_NS {v:<10} {ns:9d} ns  {ns * 1.35 / (bm * bn * k):6.2f} cyc/tile-K")
PY
fi
