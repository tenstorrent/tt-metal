#!/bin/bash
# usage: run_profile.sh <label> <cases> [--top N]   — one profiled run of every case, then the per-case summary.
set -u
ROOT=$(cd "$(dirname "$0")/../../../../../.." && pwd)
D=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT";  unset TT_METAL_DPRINT_CORES
mkdir -p "$D/logs"; echo "$2" > "$D/logs/$1.cases"
NBW_CASES="$2" timeout 1800 true --profile --run-all \
  tests/ttnn/unit_tests/operations/distributed_rms_norm/perf_experiments/noc_balanced_writes/test_noc_balanced_writes.py > "$D/logs/$1.log" 2>&1
echo "rc=$?"; grep -E "passed|failed" "$D/logs/$1.log" | tail -1
csv=$(grep -o "PROFILER CSV: .*" "$D/logs/$1.log" | tail -1 | cut -d' ' -f3)
python3 "$D/analyze.py" "$2" "$csv" "${@:3}" | tee "$D/logs/$1.summary"
