#!/usr/bin/env bash
# usage: run_bench.sh <label> <pytest -k expr | case id>   (env knobs MRS_* pass through: MRS_FF=base|0|1|2, MRS_PERF_GUARD=1)
# One fresh-cache traced run under the profiler; prints PLAN / rel_rms, op latency (max over chips) per replay, and
# the MM / MM_FUSED / FIN zones of device 3.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../../.." && pwd)"
LABEL="$1"; KEXPR="$2"
cd "$ROOT"
export MRS_PERF_TRACE=1
LOG="$HERE/results/$LABEL.log"
scripts/run_safe_pytest.sh --profile "$HERE/test_ff_perf.py" -k "$KEXPR" -s --rootdir="$HERE" > "$LOG" 2>&1
CSV=$(grep "SAFE_PYTEST: PROFILER CSV:" "$LOG" | tail -1 | sed 's/.*PROFILER CSV: //')
grep -E "PLAN|SAFE_PYTEST_RESULT|DEV [0-9] rel_rms|Error|error:|FAILED|passed|failed" "$LOG" | sort -u | cut -c1-400 | head -20
if [ -n "$CSV" ]; then
  cp "$CSV" "$HERE/results/$LABEL.ops.csv"
  cp "$(dirname "$CSV")/profile_log_device.csv" "$HERE/results/$LABEL.device.csv"
  python3 tests/ttnn/unit_tests/operations/matmul_reduce_scatter/perf_summary.py "$HERE/results/$LABEL.ops.csv"
  python3 tests/ttnn/unit_tests/operations/matmul_reduce_scatter/perf_zone_report.py "$HERE/results/$LABEL.device.csv" --device 3 2>/dev/null | grep -A6 -E "^(MM|MM_FUSED|FIN|FIN_WAIT|WR_BARRIER|ARR_WAIT) " | head -60
  gzip -f "$HERE/results/$LABEL.device.csv"
fi
