#!/bin/bash
# Usage: bench.sh <variant> [test_peel|test_guard|<path>] -- profiles one pytest run, prints DEVICE KERNEL DURATION (us) per row.
set -u
cd /localdev/kgrujcic/tt-metal-codegen
source python_env/bin/activate
here=ttnn/ttnn/operations/layer_norm_tile/perf_experiments/affine_receive_overlap
sel=${2:-test_peel}
target="$here/test_aro.py::$sel"
case "$sel" in */*) target=$sel ;; esac
tag=$1_$(basename "$sel" .py)_$(date +%H%M%S)
marker=$(mktemp)
PYTHONPATH="$PWD/$here:${PYTHONPATH:-}" LN_VARIANT=$1 scripts/run_safe_pytest.sh --run-all ${PROFILE---profile} "$target" -p swap --import-mode=prepend >"$here/logs/$tag.log" 2>&1
rc=$?
grep -E "ARO_SWAP" "$here/logs/$tag.log" | head -1
grep -E "passed|failed|SAFE_PYTEST" "$here/logs/$tag.log" | tail -2
if [ -n "${PROFILE-x}" ]; then
  csv=$(ls -t /localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
  if [ "$csv" -ot "$marker" ]; then echo "STALE CSV $csv rc=$rc"; exit 1; fi
  cp "$csv" "$here/logs/$tag.ops.csv"
  cp /localdev/kgrujcic/tt-metal/generated/profiler/.logs/profile_log_device.csv "$here/logs/$tag.dev.csv" 2>/dev/null
  echo "$1 $sel: $(awk -F',' 'NR>1 && $1 ~ /GenericOp/ {printf "%.1f ", $20/1000}' "$csv")"
fi
rm -f "$marker"
