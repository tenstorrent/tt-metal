#!/usr/bin/env bash
# Snapshot the newest profiler report into this experiment's logs/ and print the
# per-dispatch DEVICE KERNEL DURATION. Usage: collect.sh <tag>
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$D/../../../../../.." && pwd)"
TAG="$1"
DIR=$(ls -td "$REPO"/generated/profiler/reports/*/ | head -1)
cp "$DIR/profile_log_device.csv" "$D/logs/dev_${TAG}.csv"
cp "$DIR"/ops_perf_results*.csv "$D/logs/ops_${TAG}.csv"
python3 - "$D/logs/ops_${TAG}.csv" "$TAG" <<'PY'
import csv,sys,statistics
rows=list(csv.DictReader(open(sys.argv[1])))
ns=[int(r["DEVICE KERNEL DURATION [ns]"]) for r in rows]
cc=[r["CORE COUNT"] for r in rows]
print(f"{sys.argv[2]}: device_ns={ns} cores={cc}")
print(f"  measured reps (rep0 dropped per case): median {statistics.median(ns[1:]) if len(ns)>1 else ns[0]:.0f}")
PY
