#!/bin/bash
# Every attempt-3 device verdict, one line each, newest last. Read-only.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
printf '%-30s %-4s %-28s %s\n' NAME RC SUMMARY FIRST-ERROR
grep '^a3_' "$D/VERDICTS_A3.txt" | while IFS='|' read -r head sum acc err; do
    name=${head%% *}; rc=${head#* rc=}; rc=${rc%% *}
    printf '%-30s %-4s %-28s %s\n' "$name" "$rc" "$(echo "$sum" | cut -c1-28)" "$(echo "${acc}${err}" | cut -c1-90)"
done
echo
echo "pending queue items: $(grep -cvE '^\s*(#|$)' "$D/queue.txt")"
echo "in flight: $(grep -oE '^--- dequeued [A-Za-z0-9_]+' "$D/logs2/queue.out" | tail -1)"
