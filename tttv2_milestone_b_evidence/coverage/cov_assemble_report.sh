#!/bin/bash
# Append attempt 2's section to attempt 1's REPORT.md, from the fragments, in
# order. Idempotent: refuses if the marker is already there.
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
cd "$D"
grep -q '^# §A2 — attempt 2, on a live mesh' REPORT.md && { echo "§A2 already present; not appending"; exit 1; }
cat A2_SECTION_HEAD.md A2_GATE_TABLE.md A2_METHOD.md A2_AREA_MAP.md \
    A2_AREAS.md A2_L1.md A2_FINDINGS.md A2_GATE_COMMANDS.md A2_CLOSE.md >> REPORT.md
echo "appended $(wc -l < REPORT.md) lines total"
