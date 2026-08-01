#!/usr/bin/env bash
# PERF 14 — measure one K-chunk variant and print `DEVICE KERNEL DURATION [ns]` per case.
#
#   perf_experiments/kchunk/measure.sh "<label>" "<MOE_KC_CASES>"
#
# Every MOE_SWIGLU_* knob is inherited from the caller's environment:
#
#   MOE_SWIGLU_GRID=11x8 MOE_SWIGLU_GU_KCHUNKS=4 perf_experiments/kchunk/measure.sh kc4 "7168,5120,256,bf16_rm"
#
# TT_METAL_PROFILER_DIR is forced to a PRIVATE directory under this experiment: the shared
# generated/profiler tree is raced by the sibling part-optimizers on this device, and the loser of
# that race silently reports someone else's CSV. The CSV path is taken from tracy's own line,
# NEVER by mtime.
set -u
LABEL="${1:-run}"
CASES="${2:-7168,5120,256,bf16_rm}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../../.." && pwd)"
cd "$ROOT" || exit 1

export TT_METAL_PROFILER_DIR="$HERE/profiler_artifacts"
mkdir -p "$TT_METAL_PROFILER_DIR"

LOG="$HERE/profiler_artifacts/measure_$$.log"
MOE_KC_CASES="$CASES" timeout 3000 scripts/run_safe_pytest.sh --run-all --profile \
    ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/kchunk/test_kchunk.py \
    >"$LOG" 2>&1
RC=$?

CSV=$(grep "OPs csv generated at:" "$LOG" 2>/dev/null | tail -1 | sed "s/.*generated at: //")
grep -h "\[kchunk\]" "$LOG" || true
if [ "$RC" != "0" ] || [ -z "$CSV" ] || [ ! -f "$CSV" ]; then
    echo "$LABEL: FAILED (rc=$RC, csv='$CSV') — see $LOG"
    grep -E "FAILED|Error|error:|assert|GU_KCHUNKS" "$LOG" | head -25
    tail -30 "$LOG"
    exit 1
fi

python3 - "$CSV" "$LABEL" "$CASES" <<'PYEOF'
import csv, sys
csv_path, label, cases = sys.argv[1], sys.argv[2], sys.argv[3]
if cases == "guard":
    sys.path.insert(0, "ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/kchunk")
    from test_kchunk import GUARD_SET

    cases = GUARD_SET
names = [c.strip() for c in cases.split(";") if c.strip()]
ns = []
for r in csv.DictReader(open(csv_path)):
    if r.get("OP CODE") == "GenericOpDeviceOperation":
        ns.append((int(r["GLOBAL CALL COUNT"]), int(r["DEVICE KERNEL DURATION [ns]"]), int(r["CORE COUNT"])))
ns.sort()
for name, (_, dur, cores) in zip(names, ns):
    print(f"{label:24s} {name:28s} {dur:9,d} ns   cores={cores}")
print(f"# csv: {csv_path}")
PYEOF
