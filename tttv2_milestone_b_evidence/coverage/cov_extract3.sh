#!/bin/bash
# One compact row per log: commit, exit code, pytest summary, wall clock, and the
# test's own bracketed marker lines. Read-only; touches no device.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
for n in "$@"; do
    f="$D/logs2/$n.log"; [ -f "$f" ] || f="$D/logs3/$n.log"
    [ -f "$f" ] || { echo "$n: NO LOG"; continue; }
    printf '%s | commit=%s | %s | %s\n' "$n" \
        "$(grep -am1 '^commit:' "$f" | cut -d' ' -f2 | cut -c1-11)" \
        "$(grep -aoE '^exit=[0-9]+' "$f" | tail -1)" \
        "$(grep -aoE '[0-9]+ (passed|failed|error|skipped)[a-z]*(, [0-9]+ [a-z]+)* in [0-9.]+s( \(0:[0-9:]+\))?' "$f" | tail -1)"
done
