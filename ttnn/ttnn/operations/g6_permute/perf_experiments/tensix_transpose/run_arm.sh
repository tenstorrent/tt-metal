#!/usr/bin/env bash
# Usage: run_arm.sh <arm> <shape_id> [op_code_substring]   (one profiled process per arm)
set -u
ROOT=/localdev/pmilojevic/campaign/bh-lb-18/slots/chip3
cd "$ROOT"
export TT_METAL_HOME=$ROOT PYTHONPATH=$ROOT
source python_env/bin/activate
ARM=$1; SHAPE=$2; CODE=${3:-GenericOp}
for attempt in 1 2 3; do
  OUT=$(timeout 900 scripts/run_safe_pytest.sh --profile --no-py-zones \
    "tests/ttnn/unit_tests/operations/g6_permute/perf1/tensix_transpose/test_tt_bench.py::test_bench[arm=${ARM}-shape_id=${SHAPE}]" -s 2>&1)
  echo "$OUT" | grep -E "TTX_PLAN|TTX_ARM|passed|failed|SAFE_PYTEST_RESULT|AssertionError" | head -8
  CSV=$(echo "$OUT" | grep "SAFE_PYTEST: PROFILER CSV:" | awk '{print $NF}')
  if [[ -n "$CSV" ]]; then
    echo "CSV $CSV"
    RES=$(python3 ttnn/ttnn/operations/g6_permute/perf_experiments/tensix_transpose/bench_parse.py "$CSV" "$CODE" 2>&1)
    if echo "$RES" | grep -q "median="; then
      echo "$RES" | sed "s/^/$ARM $SHAPE /"
      exit 0
    fi
    echo "parse failed ($RES), retrying"
    continue
  fi
  echo "no CSV (attempt $attempt), retrying"
done
