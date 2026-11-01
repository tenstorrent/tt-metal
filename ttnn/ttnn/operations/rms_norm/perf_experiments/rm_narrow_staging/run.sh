#!/bin/bash
# usage: RNS_CELLS=... RNS_VARIANTS=... run.sh <tag>
# pytest cannot collect a test inside the ttnn/ttnn package (its Package node re-imports ttnn/__init__.py), so the
# bench runs through a symlink in /tmp with the repo-root conftest loaded as a plugin.
set -u
ROOT=/localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/rms_norm/perf_experiments/rm_narrow_staging
cd $ROOT && source python_env/bin/activate
mkdir -p /tmp/rns_bench $D/runs && ln -sf $D/bench_rm_narrow_staging.py /tmp/rns_bench/test_rns_bench.py
out=$(scripts/run_safe_pytest.sh --profile /tmp/rns_bench/test_rns_bench.py -s -p conftest 2>&1)
csv=$(echo "$out" | sed -n 's/^SAFE_PYTEST: PROFILER CSV: //p')
[ -n "$csv" ] && cp "$csv" $D/runs/ops_$1.csv && cp generated/profiler/.logs/profile_log_device.csv $D/runs/dev_$1.csv
echo "$out" > $D/runs/log_$1.txt
echo "$out" | grep -E "RNS_|passed|failed|Error|SAFE_PYTEST_RESULT|triage" | grep -v "^SAFE_PYTEST: env" | head -80
[ -n "$csv" ] && python3 - "$D/runs/ops_$1.csv" "$D/runs/log_$1.txt" <<'PY'
import csv, sys, re
rows = list(csv.DictReader(open(sys.argv[1])))  # this Tracy CSV has no OP CODE column: every row is a dispatch
log = open(sys.argv[2]).read()
log = log[log.find("starting the real pass"):] if "starting the real pass" in log else log
order = re.findall(r"RNS_ORDER (\S+) (\S+)", log)
print(f"generic rows {len(rows)} order {len(order)}")
for (c, v), r in zip(order, rows):
    print(f"{c:24s} {v:10s} {int(r['DEVICE KERNEL DURATION [ns]']):>10d}")
PY
