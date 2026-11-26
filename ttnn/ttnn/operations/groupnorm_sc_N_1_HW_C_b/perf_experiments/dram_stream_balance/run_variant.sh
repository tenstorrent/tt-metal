#!/usr/bin/env bash
# usage: run_variant.sh <variant[,variant...]> <pytest -k expr>
# One profiled pytest session over variants x selected cases (one device-lock acquisition). Prints the
# DEVICE KERNEL DURATION per (variant, case) in dispatch order, plus the report dir (profile_log_device.csv inside;
# run host IDs follow the same order). Env: DSB_CHECK=1 adds the PCC / rel-RMS gate.
set -u
cd "$(dirname "$0")/../../../../../.."
D=ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/perf_experiments/dram_stream_balance
V="$1"; K="$2"
OUT=$(DSB_VARIANT="$V" timeout 590 scripts/run_safe_pytest.sh --run-all --profile --no-py-zones "$D/test_dsb_bench.py" -k "$K" --import-mode=prepend 2>&1)
echo "$OUT" > $D/last_run.out
echo "$OUT" | grep -v "riscv-tt-elf-g++" | grep -E "DSB_CHECK|DSB_PLAN|passed|failed|Error|SAFE_PYTEST_RESULT|TIMEOUT|hang|Caught SIG|lock acquired" | grep -v "INFO" | tail -40
CSV=$(echo "$OUT" | grep "PROFILER CSV" | sed 's/.*PROFILER CSV: //')
RUNS=$(echo "$OUT" | grep -o "DSB_RUN [^ ]* [^ ]*" | sed 's/DSB_RUN //' | tr ' ' ':' | tr '\n' ' ')
if [[ -n "$CSV" ]]; then
  python3 - "$CSV" "$RUNS" << 'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["OP CODE"].startswith("GenericOp")]
runs = sys.argv[2].split()
print(f"report {sys.argv[1].rsplit('/', 1)[0]}")
for i, r in enumerate(rows):
    lab = runs[i] if i < len(runs) else "?"
    print(f"  {lab:60s} ns {r['DEVICE KERNEL DURATION [ns]']:>10s}  cores {r['CORE COUNT']}  call {r['GLOBAL CALL COUNT']}")
PY
  echo "$V | $K | $(dirname "$CSV")" >> $D/runs.log
fi
