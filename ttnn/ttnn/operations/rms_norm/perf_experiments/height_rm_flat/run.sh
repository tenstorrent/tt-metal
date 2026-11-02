#!/bin/bash
# usage: HRF_CELLS=... HRF_VARIANTS=... run.sh <tag> [--dev]
# pytest cannot collect a test inside the ttnn/ttnn package, so the bench runs through a symlink in /tmp with the
# repo-root conftest loaded as a plugin.
set -u
ROOT=/localdev/adamjanovic/2026_11_01/1349_adamjanovic_ab-tilize-run-scoreboard-catalog4/clones/rms_norm_run1/tt-metal
D=$ROOT/ttnn/ttnn/operations/rms_norm/perf_experiments/height_rm_flat
cd $ROOT && source python_env/bin/activate
mkdir -p /tmp/hrf_bench $D/runs && ln -sf $D/bench_height_rm_flat.py /tmp/hrf_bench/test_hrf_bench.py
PROF="--profile"; [ "${2:-}" = "--dev" ] && PROF="--dev"
out=$(scripts/run_safe_pytest.sh $PROF --run-all /tmp/hrf_bench/test_hrf_bench.py -s -p conftest 2>&1)
csv=$(echo "$out" | sed -n 's/^SAFE_PYTEST: PROFILER CSV: //p')
[ -n "$csv" ] && cp "$csv" $D/runs/ops_$1.csv && cp generated/profiler/.logs/profile_log_device.csv $D/runs/dev_$1.csv
echo "$out" > $D/runs/log_$1.txt
echo "$out" | grep -E "HRF_GEOM|HRF_PREC|passed|failed|Error|error:|SAFE_PYTEST_RESULT|triage" | grep -v "^SAFE_PYTEST: env" | head -120
[ -n "$csv" ] && python3 $D/table.py "$D/runs/ops_$1.csv" "$D/runs/log_$1.txt"
