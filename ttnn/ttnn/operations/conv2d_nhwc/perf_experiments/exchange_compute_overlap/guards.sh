#!/bin/bash
# guards.sh <cases...>: one fresh profiled run per (case, variant): variant baseline vs in-place candidate.
cd "$(dirname "$0")"
for c in "$@"; do
  for v in base:ECO_INPLACE=0 cand:ECO_INPLACE=1; do
    l=g_${c}_${v%%:*}
    out=$(HF_CASE=$c HF_SET=${v#*:} ./run.sh $l 2>&1)
    echo "$c ${v%%:*} $(echo "$out" | grep -v nan | grep -oE 'pcc=[0-9.]+ rel_rms=[0-9.]+' | head -1) $(echo "$out" | grep -E 'TRISC_1' | awk '{print $3}') $(echo "$out" | grep -E 'FAILED|Timeout' | head -1)"
  done
done
