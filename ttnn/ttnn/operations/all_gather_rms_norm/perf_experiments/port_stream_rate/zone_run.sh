#!/bin/bash
# usage: zone_run.sh <tag> [ENV=VAL ...]  -> results/z_<tag>.csv + per-port breakdown (per-packet zones on)
cd "$(dirname "$0")/../../../../../.."
D=ttnn/ttnn/operations/all_gather_rms_norm/perf_experiments/port_stream_rate
tag=$1; shift
env "$@" AGRMS_PROFILE_ZONES=1 TT_METAL_DEVICE_PROFILER=1 PSR_ZONE_OUT=$PWD/$D/results/z_$tag.csv \
    scripts/tt-probe.sh all_gather_rms_norm < $D/zone_probe.py 2>&1 | grep -E "GOLDEN_OK|PROBE_DONE|TT_PROBE_RESULT|Error|error:" | head
source python_env/bin/activate
python3 $D/zparse.py $D/results/z_$tag.csv
