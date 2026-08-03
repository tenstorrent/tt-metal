#!/usr/bin/env bash
# Knob search at the GRADED cells, on the op's designed call site.
#
#   perf_experiments/knob_search.sh "<label>=<VAR=val ...>" ["<label2>=..."] ...
#
# Baseline is the shipped default at 88 cores (11x8, the grid the graded targets were taken on) with
# ND-SHARDED weights (the placement PERF 12 designs for — `DRAM_MEMORY_CONFIG` weights silently take
# the uncoalesced path, so an interleaved harness understates the op by up to 11 %).
#
# Uses the seqlen-sweep harness because it is the only one that can place sharded weights and it
# asserts the placement against the reader's own predicate. --no-precompile is REQUIRED: the warmup
# OOMs on multi-case sessions at capacity 5120 and the SIGKILL wedges the board.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT" || exit 1
OUT="${KNOB_OUT:-/tmp/moe_knob}"
mkdir -p "$OUT"
COUNTS="${KNOB_COUNTS:-128,256,512}"
FMT="${KNOB_FMT:-bf16_rm}"
REPS="${KNOB_REPS:-3}"
TEST=tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_seqlen_sweep.py

for SPEC in "$@"; do
    LABEL="${SPEC%%=*}"
    ENVS="${SPEC#*=}"
    [ "$ENVS" = "$LABEL" ] && ENVS=""
    LOG="$OUT/log_$LABEL.log"
    MAN="$OUT/man_$LABEL.json"
    # shellcheck disable=SC2086
    env $ENVS MOE_SWIGLU_GRID=11x8 MOE_SWEEP_COUNTS="$COUNTS" MOE_SWEEP_FORMATS="$FMT" \
        MOE_SWEEP_WPLACE=nd_shard MOE_SWEEP_REPS="$REPS" MOE_SWEEP_WARMUP=2 \
        MOE_SWEEP_EMB=7168 MOE_SWEEP_CAPACITY=5120 MOE_SWEEP_MANIFEST="$MAN" \
        timeout 1200 scripts/run_safe_pytest.sh --profile --no-precompile "$TEST" >"$LOG" 2>&1
    RC=$?
    CSV=$(grep "OPs csv generated at:" "$LOG" | tail -1 | sed 's/.*generated at: //')
    if [ "$RC" != "0" ] || [ ! -f "$CSV" ]; then
        echo "$LABEL: FAILED (rc=$RC) — $LOG"
        grep -E "FAILED|Error|error|assert" "$LOG" | head -5
        continue
    fi
    python3 ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/parse_seqlen_sweep.py \
        "$OUT/res_$LABEL" "$CSV" "$MAN" >"$OUT/parse_$LABEL.txt" 2>&1
    python3 - "$OUT/res_$LABEL.json" "$LABEL" "$ENVS" <<'PYEOF'
import json, sys
recs = json.load(open(sys.argv[1]))["points"]
cells = " | ".join(f"{r['count']}:{r['us_median']:8.2f}" for r in sorted(recs, key=lambda r: r["count"]))
tot = sum(r["ns_median"] for r in recs)
print(f"{sys.argv[2]:<16} {cells} | sum {tot/1e3:9.2f} us   [{sys.argv[3] or 'shipped defaults'}]")
PYEOF
done
