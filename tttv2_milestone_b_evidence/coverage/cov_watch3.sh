#!/bin/bash
# Attempt 3: as each queue item finishes, append one machine-written line to
# VERDICTS_A3.txt: name, rc, pytest summary, and the first distinctive error.
# Read-only with respect to the device; never signals anything.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
V="$D/VERDICTS_A3.txt"
touch "$V"
while true; do
    while read -r name rc; do
        grep -q "^$name " "$V" 2>/dev/null && continue
        L="$D/logs2/$name.log"
        sum=$(grep -oE '[0-9]+ (passed|failed|error)[a-z]*(, [0-9]+ (passed|failed|error)[a-z]*)*' "$L" 2>/dev/null | tail -1)
        err=$(grep -hoE 'TT_(FATAL|THROW)[^|]*|Statically allocated circular buffers[^"]{0,120}|assert [^\n]{0,110}|E +[A-Za-z_]*Error[^\n]{0,110}' "$L" 2>/dev/null | head -1 | cut -c1-200)
        acc=$(grep -hoE 'top-?[15][^|,]{0,60}' "$L" 2>/dev/null | head -4 | tr '\n' ' ')
        printf '%s rc=%s | %s | %s | %s\n' "$name" "$rc" "${sum:-NO-SUMMARY}" "${acc:- }" "${err:- }" >> "$V"
    done < <(grep -oE '^--- [A-Za-z0-9_]+ rc=[0-9]+' "$D/logs2/queue.out" 2>/dev/null | awk '{print $2, $3}' | sed 's/rc=//')
    sleep 45
done
