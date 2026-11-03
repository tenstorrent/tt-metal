#!/bin/bash
# Usage: MOE_CASES=focus ablate.sh "name:DEF1,DEF2" ...  -> "<variant> <case> <ns>"
here=$(cd "$(dirname "$0")" && pwd)
for v in "$@"; do
    name=${v%%:*}; defs=${v#*:}
    out=$(MOE_KERNEL_DIR=$here/kernels_ablate MOE_DEFINES=$defs MOE_CHECK=0 "$here/bench.sh" | grep -v "^log:\|passed\|failed")
    while read -r line; do echo "$name $line"; done <<<"$out"
done
