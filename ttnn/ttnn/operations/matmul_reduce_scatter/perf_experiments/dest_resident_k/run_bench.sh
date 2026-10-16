#!/usr/bin/env bash
# usage: run_bench.sh <label> <case id, e.g. M640_K2048_N7168_sd-1>   (env knobs MRS_* pass through)
# Runs the experiment perf harness traced under the profiler, then summarizes:
# per-device DEVICE KERNEL DURATION of the measured generic-op rows, and the MM / FIN zones of device 3.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../../.." && pwd)"
LABEL="$1"; KEXPR="$2"
cd "$ROOT"
export MRS_PERF_TRACE=1
LOG="$HERE/results/$LABEL.log"
scripts/run_safe_pytest.sh --profile "$HERE/test_drk_perf.py::test_perf_case[$KEXPR]" -s --rootdir="$HERE" > "$LOG" 2>&1
CSV=$(grep "SAFE_PYTEST: PROFILER CSV:" "$LOG" | tail -1 | sed 's/.*PROFILER CSV: //')
grep -E "PLAN|SAFE_PYTEST_RESULT|DEV [0-9] rel_rms|Error|error:" "$LOG" | sort -u | cut -c1-400 | head -12
if [ -n "$CSV" ]; then
  cp "$CSV" "$HERE/results/$LABEL.ops.csv"
  cp "$(dirname "$CSV")/profile_log_device.csv" "$HERE/results/$LABEL.device.csv"
  python3 "$HERE/summarize.py" "$HERE/results/$LABEL.ops.csv"
  python3 tests/ttnn/unit_tests/operations/matmul_reduce_scatter/perf_zone_report.py "$HERE/results/$LABEL.device.csv" --device 3 --all 2>/dev/null | grep -A1 -E "^(MM|FIN) " | head -24
  gzip -f "$HERE/results/$LABEL.device.csv"
fi
