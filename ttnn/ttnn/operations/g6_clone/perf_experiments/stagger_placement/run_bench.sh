#!/usr/bin/env bash
# Placement lane runs: run_bench.sh <logdir> <case>:<place|base|stock> ...; one bench process per item.
# place = "k,r,ppc" written k.r.ppc on the command line (e.g. F10:4.8.1), or "rule".
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
set +u; source python_env/bin/activate; set -u
export TT_METAL_HOME=$PWD
E=ttnn/ttnn/operations/g6_clone/perf_experiments/stagger_placement
# A bench file under ttnn/ re-imports ttnn under pytest: run a copy from the git-ignored generated/.
R=generated/meta/g6_clone/lanes/perf1_stagger_placement; mkdir -p $R; cp $E/bench_cases.py $R/
export PLACE_ROOT=$PWD/$E
L=$1; shift; mkdir -p $L
for item in "$@"; do
  IFS=: read -r c v <<< "$item"
  impl=new; place=""
  case $v in base) ;; stock) impl=stock ;; *) place=${v//./,} ;; esac
  tag=${c}_${v}
  for try in 1 2 3; do
    PLACE=$place BENCH_IMPL=$impl BENCH_CASES=$c TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile --no-py-zones $R/bench_cases.py::test_bench > $L/$tag.log 2>&1
    csv=$(grep -a 'SAFE_PYTEST: PROFILER CSV:' $L/$tag.log | tail -1 | sed 's/.*PROFILER CSV: //')
    [ -n "$csv" ] && break
  done
  ok=$(grep -ac '^BENCH_OK' $L/$tag.log)
  cp generated/profiler/.logs/profile_log_device.csv $L/$tag.device.csv
  python3 .claude/skills/tooling/scripts/ops_csv.py summary "$csv" --skip-first 1 > $L/$tag.summary.txt 2>&1
  ts=$(python3 $E/true_span.py $L/$tag.device.csv 1 | sed 's/ cores_xy=.*//')
  echo "$tag ok=$ok $(sed -n 3p $L/$tag.summary.txt | awk '{print "median",$5,"cores",$3}') $ts $(grep -a PLACE_INSTALLED $L/$tag.log | head -1)"
done
