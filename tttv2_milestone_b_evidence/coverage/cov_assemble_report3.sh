#!/bin/bash
# Append attempt 3's section to REPORT.md, from the fragments, in order.
# Idempotent: refuses if the marker is already there.
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
cd "$D"
grep -q '^# §A3 — attempt 3, the completing pass' REPORT.md && { echo "§A3 already present; not appending"; exit 1; }
for f in A3_SECTION_HEAD.md A3_GATE_TABLE.md A3_METHOD.md A3_AREA_MAP.md A3_AREAS.md \
         A3_L1.md A3_FINDINGS.md A3_GATE_COMMANDS.md A3_CLOSE.md; do
    [ -f "$f" ] || { echo "missing fragment: $f"; exit 2; }
    grep -qE '@@[A-Z_]+@@' "$f" && { echo "unresolved @@PLACEHOLDER@@ in $f - refusing to assemble"; exit 3; }
done
cat A3_SECTION_HEAD.md A3_GATE_TABLE.md A3_METHOD.md A3_AREA_MAP.md A3_AREAS.md \
    A3_L1.md A3_FINDINGS.md A3_GATE_COMMANDS.md A3_CLOSE.md >> REPORT.md
echo "appended; REPORT.md is now $(wc -l < REPORT.md) lines"
