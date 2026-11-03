#!/bin/bash
# Usage: run.sh <defines-or-"base"> [cases]  -> bench.sh with this dir's kernels (MOE_CHECK on).
here="$(cd "$(dirname "$0")" && pwd)"
d=$1; [ "$d" = base ] && d=""
MOE_KERNEL_DIR="$here/kernels" MOE_DEFINES="$d" MOE_CASES="${2:-focus}" MOE_CHECK=${MOE_CHECK:-1} \
    "$here/../breakdown_r2/bench.sh"
