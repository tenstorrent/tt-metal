#!/usr/bin/env bash
# One log, always the same path, with a HEARTBEAT so a hang is visible to a human.
#
#   Terminal A (you):   tail -f /home/nwoodall/tt-metal/generated/current.log
#   Terminal B (or me): ./watch_run.sh "LABEL" [extra pytest args]
#
# Env for the run is taken from the caller, e.g.
#   DIFFVAE_S5_GNA_STRIDE=6,8,4 ./watch_run.sh "fair stride"
#
# Every 15s a [HEARTBEAT] line reports elapsed time and, crucially, how long the RUN has been
# silent. quiet=0-20s is normal work; quiet climbing past ~60s while the run is still alive means
# it is hung, and you can say so without waiting out the timeout. Banners mark start and result.
set -uo pipefail
cd /home/nwoodall/tt-metal

LABEL="${1:-run}"; shift || true
LOG=generated/current.log
RAW=generated/current.raw.log
TIMEOUT="${WATCH_TIMEOUT:-900}"

export LTX25_ROOT="${LTX25_ROOT:-$HOME/.cache/ltx-checkpoints/ltx-2.5}"
export DIFFVAE_CHECKPOINT="${DIFFVAE_CHECKPOINT:-$LTX25_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors}"

mkdir -p generated
: > "$RAW"; : > "$LOG"   # truncate in place so an existing tail -f keeps working

{
  echo "================================================================"
  echo "RUN: $LABEL"
  echo "started : $(date -Is)   timeout: ${TIMEOUT}s"
  echo "stride  : ${DIFFVAE_S5_GNA_STRIDE:-<default 1,1,1>}   backend: ${DIFFVAE_STAGE5_BACKEND:-<default op_sp_w_sharded>}"
  echo "TP_HEADS: ${DIFFVAE_TP_HEADS:-1}   latent_T: ${DIFFVAE_LATENT_T:-19}   slab: ${DIFFVAE_SLAB_FRAMES:-73}"
  echo "================================================================"
} >> "$LOG"

stdbuf -oL -eL timeout "$TIMEOUT" bash models/tt_dit/experimental/scripts/run_ltx25_diffvae.sh "$@" > "$RAW" 2>&1 &
RUN_PID=$!

tail -n +1 -f "$RAW" >> "$LOG" &
TAIL_PID=$!

START=$(date +%s)
( while kill -0 "$RUN_PID" 2>/dev/null; do
    sleep 15
    kill -0 "$RUN_PID" 2>/dev/null || break
    NOW=$(date +%s)
    QUIET=$(( NOW - $(stat -c %Y "$RAW") ))
    NOTE=""
    [ "$QUIET" -ge 60 ]  && NOTE="  <-- SILENT ${QUIET}s, likely HUNG"
    [ "$QUIET" -ge 180 ] && NOTE="  <-- SILENT ${QUIET}s, HUNG (say so and I will kill it)"
    echo "[HEARTBEAT] elapsed=$(( NOW - START ))s quiet=${QUIET}s${NOTE}" >> "$LOG"
  done ) &
BEAT_PID=$!

wait "$RUN_PID"; STATUS=$?
sleep 1; kill "$BEAT_PID" "$TAIL_PID" 2>/dev/null

{
  echo "================================================================"
  case "$STATUS" in
    0)   echo "RESULT: PASSED  ($LABEL)" ;;
    124) echo "RESULT: TIMED OUT after ${TIMEOUT}s -- HUNG  ($LABEL)" ;;
    *)   echo "RESULT: FAILED exit $STATUS  ($LABEL)" ;;
  esac
  grep -E "decode W-SP|^decode TOTAL" "$RAW" | tail -3
  echo "finished: $(date -Is)"
  echo "================================================================"
} >> "$LOG"
exit "$STATUS"
