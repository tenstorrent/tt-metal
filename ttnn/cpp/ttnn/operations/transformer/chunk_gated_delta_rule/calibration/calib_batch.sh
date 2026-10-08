#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# calib_batch.sh <rows file> [<rows file> ...]: one Tracy capture per row "label script args...", under the device
# lock and the private JIT cache (calib_env.sh). A row whose capture already holds a ChunkGdn op median is skipped,
# so a batch can be re-run to fill in failures. The run itself (not the wait for the lock) is limited to 8 minutes:
# a run that blocks longer is killed, the board is reset and the row retried once. Prints the label, the args and
# the op medians as rows complete. Scripts are resolved in this directory unless given as a path; CALIB_ITERS
# overrides the rows' --iters.
set -uo pipefail
source "$(dirname "$(readlink -f "$0")")/calib_env.sh"
cd "$TT_METAL_HOME"
for rows in "$@"; do
  while read -r label script args; do
    { [ -z "$label" ] || [ "${label:0:1}" = "#" ]; } && continue
    out=$CALIB_OUT/$label
    if [ -s "$out/ops_summary.txt" ] && grep -q ChunkGdn "$out/ops_summary.txt"; then echo "$label  (exists)"; continue; fi
    case "$script" in /*) s=$script ;; *) s=$CALIB_DIR/$script ;; esac
    [ -n "${CALIB_ITERS:-}" ] && args="$(sed -E 's/--iters [0-9]+/--iters '"$CALIB_ITERS"'/' <<<"$args")"
    mkdir -p "$out"
    for attempt in 1 2; do
      echo "== $(date '+%F %T') $label :: $s $args (tree $(git rev-parse --short HEAD 2>/dev/null))" >> "$out/run.log"
      flock "$CALIB_LOCK" timeout -k 15 480 python -m tracy -r -p -o "$out" --op-support-count 4000 "$s" $args >> "$out/run.log" 2>&1; rc=$?
      echo "tracy exit $rc" >> "$out/run.log"
      if [ $rc -eq 124 ]; then echo "$label  TIMEOUT (attempt $attempt): resetting the board"; flock "$CALIB_LOCK" tt-smi -r > /dev/null 2>&1; sleep 5; continue; fi
      break
    done
    python "$CALIB_DIR/calib_ops.py" "$out" --json "$out/ops_summary.json" > "$out/ops_summary.txt" 2>/dev/null
    summ=$(grep -h "ChunkGdn" "$out/ops_summary.txt" 2>/dev/null | awk '{printf "%s %s | ", $1, $4}')
    err=$(grep -h "FATAL\|Error\|TT_THROW\|Traceback" "$out/run.log" 2>/dev/null | head -1 | cut -c1-140)
    echo "$(date '+%T') $label  $args  => ${summ:-FAILED} ${err}"
  done < "$rows"
  echo "batch done: $rows"
done
