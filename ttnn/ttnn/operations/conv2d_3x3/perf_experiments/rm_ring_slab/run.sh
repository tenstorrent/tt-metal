#!/bin/bash
# usage: run.sh <tag> <cases> <variants>   e.g. run.sh t1 idx41 base,ring
# Profiles test_rm_ring.py (median of 3 signposted calls per variant via cmp_windows.py); appends to perf.txt.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
D=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/rm_ring_slab
tag=$1; cases=$2; variants=$3
out=$(CONV2D_3X3_ABLATE=${ABLATE:-} RMR_TAG=$tag RMR_CASES=$cases RMR_VARIANTS=$variants timeout 3000 scripts/run_safe_pytest.sh --profile --run-all $D/test_rm_ring.py -p no:cacheprovider --import-mode=prepend 2>&1)
echo "$out" | grep -E "RMR:" | sed 's/"shape[^]]*], //'
echo "$out" | grep -E " passed| failed|SAFE_PYTEST_RESULT|HANG|triage" | tail -3
csv=$(echo "$out" | grep "PROFILER CSV:" | sed 's/.*CSV: //')
mkdir -p $D/runs
# Zones: generated/profiler/.logs/profile_log_device.csv (zone_report.py) until the next profiled run.
[ -n "$csv" ] && gzip -c "$csv" > $D/runs/${tag}_ops.csv.gz
[ -n "$csv" ] && python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py "$csv" | awk -v t="$tag" '{print t"\t"$0}' | tee -a $D/perf.txt
