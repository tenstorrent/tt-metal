#!/usr/bin/env bash
# Snapshot the newest profiler report into this experiment's logs/ and pair each
# DEVICE KERNEL DURATION row with the (mode, rep) that dispatched it.
# Usage: collect.sh <tag>
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$D/../../../../../.." && pwd)"
TAG="$1"
DIR=$(ls -td "$REPO"/generated/profiler/reports/*/ | head -1)
cp "$DIR/profile_log_device.csv" "$D/logs/dev_${TAG}.csv"
cp "$DIR"/ops_perf_results*.csv "$D/logs/ops_${TAG}.csv"
python3 "$D/summarize.py" --ops "$D/logs/ops_${TAG}.csv" --dispatches "$D/logs/dispatches_${TAG}.json" | tee "$D/logs/summary_${TAG}.txt"
