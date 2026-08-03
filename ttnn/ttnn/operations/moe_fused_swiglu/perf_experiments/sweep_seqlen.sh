#!/usr/bin/env bash
# Run the sequence-length sweep in CHUNKS and collect one (report dir, manifest) pair per chunk.
#
#   perf_experiments/sweep_seqlen.sh <format> <counts_per_chunk> [emb] [capacity] [step] [reps]
#
# Chunking is forced by tracy's host-side report generator, not by the device: it needs ~50 MB RSS
# per profiled dispatch, so a whole 160-point sweep in one session is OOM-killed after the device
# work has already succeeded. Each chunk is its own pytest session and its own profiler report.
#
# Emits `PAIR <report_dir> <manifest>` per chunk on stdout; feed them all to parse_seqlen_sweep.py.
set -u
FORMAT="${1:-bf16_rm}"
PER_CHUNK="${2:-48}"
EMB="${3:-7168}"
CAPACITY="${4:-5120}"
STEP="${5:-32}"
REPS="${6:-3}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT" || exit 1
OUT="${MOE_SWEEP_OUTDIR:-/tmp/moe_seqlen}"
mkdir -p "$OUT"
PAIRS="$OUT/pairs_${FORMAT}.txt"
: >"$PAIRS"

CHUNK_SPAN=$((PER_CHUNK * STEP))
CHUNK=0
LO=$STEP
while [ "$LO" -le "$CAPACITY" ]; do
    HI=$((LO + CHUNK_SPAN - STEP))
    [ "$HI" -gt "$CAPACITY" ] && HI=$CAPACITY
    CHUNK=$((CHUNK + 1))
    MANIFEST="$OUT/manifest_${FORMAT}_${LO}_${HI}.json"
    LOG="$OUT/log_${FORMAT}_${LO}_${HI}.log"
    echo "=== chunk $CHUNK: $FORMAT counts $LO..$HI (reps=$REPS) ==="

    MOE_SWEEP_FORMATS="$FORMAT" MOE_SWEEP_EMB="$EMB" MOE_SWEEP_CAPACITY="$CAPACITY" \
        MOE_SWEEP_STEP="$STEP" MOE_SWEEP_REPS="$REPS" MOE_SWEEP_LO="$LO" MOE_SWEEP_HI="$HI" \
        MOE_SWEEP_MANIFEST="$MANIFEST" \
        timeout 1800 scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_seqlen_sweep.py \
        >"$LOG" 2>&1
    RC=$?

    # tracy prints the authoritative CSV path; run_safe_pytest's own PROFILER CSV line looks in the
    # clone while tracy writes under TT_METAL_HOME, so it is often absent. Never pick "newest report
    # dir" by mtime — that silently attributes another agent's concurrent run to this chunk.
    CSV=$(grep "OPs csv generated at:" "$LOG" 2>/dev/null | tail -1 | sed "s/.*generated at: //")
    [ -f "$CSV" ] || CSV=$(grep -m1 "SAFE_PYTEST: PROFILER CSV:" "$LOG" 2>/dev/null | sed "s/.*PROFILER CSV: //")
    if [ "$RC" != "0" ] || [ ! -f "$CSV" ]; then
        echo "chunk $CHUNK FAILED (rc=$RC, csv='$CSV') — see $LOG"
        tail -20 "$LOG"
        exit 1
    fi
    echo "PAIR $CSV $MANIFEST"
    echo "$CSV $MANIFEST" >>"$PAIRS"
    LO=$((HI + STEP))
done
echo "pairs written to $PAIRS"
