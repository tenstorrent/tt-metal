#!/bin/bash
# usage: sweep.sh shape...  -> baseline (op today) vs candidate (split rule + mode 2) per shape
CAND='{"SEG_READ_MODE": 2, "MIN_SEG_BYTES": 1024, "SMALL_SPLIT_MAX_CORES": 16, "ROW_COST_TILES": 10}'
for sh in "$@"; do
  "$(dirname "$0")/bench.sh" $sh '{}'
  "$(dirname "$0")/bench.sh" $sh "$CAND"
done
