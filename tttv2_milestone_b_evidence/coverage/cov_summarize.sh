#!/bin/bash
# One line per device log: verdicts, the session summary, any printed measurement.
set -u
D="$(cd "$(dirname "$0")" && pwd)/logs2"
for f in "$D"/a2_*.log; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .log)
    summary=$(grep -oE '[0-9]+ (passed|failed|error|skipped)[^,)]*' "$f" | tr '\n' ',' | sed 's/,$//')
    exitc=$(grep -oE '^exit=[0-9]+' "$f" | tail -1)
    p=$(grep -cE '^PASSED' "$f"); fl=$(grep -cE '^FAILED' "$f"); s=$(grep -cE '^SKIPPED' "$f")
    printf '%-42s P=%-3s F=%-3s S=%-3s %-22s %s\n' "$name" "$p" "$fl" "$s" "${exitc:-running}" "${summary:-}"
    grep -hoE '\[(accuracy|vocab|placement|stage|demo)\][^\n]*' "$f" | sed 's/^/      /' | head -12
    grep -hoE '^(FAILED|ERROR) [^ ]+' "$f" | sed 's/^/      /' | head -12
done
