#!/usr/bin/env bash
# bank_local_small runs: run.sh <logdir> <case>:<tag>[:ENV=VAL[,ENV=VAL...]] ...; one bench process per item.
# Example: run.sh logs/r1 Q4:base Q4:ba:BLS_VARIANT=ba F10:ba_e1:BLS_VARIANT=ba,BLS_ORDER=early,BLS_CPB=1
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
set +u; source python_env/bin/activate; set -u
export TT_METAL_HOME=$PWD
E=$PWD/ttnn/ttnn/operations/g6_clone/perf_experiments/bank_local_small
# A bench file under ttnn/ re-imports ttnn under pytest: run a copy from the git-ignored generated/.
R=generated/meta/g6_clone/lanes/perf1_bank_local_small; mkdir -p $R; cp $E/bench_cases.py $R/
export BLS_DIR=$E
L=$1; shift; mkdir -p $L
for item in "$@"; do
  IFS=: read -r c tag envs <<< "$item"
  vars=(); [ -n "${envs:-}" ] && IFS=, read -ra vars <<< "$envs"
  for try in 1 2 3; do
    env "${vars[@]}" BENCH_CASES=$c TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile --no-py-zones \
        $R/bench_cases.py::test_bench > $L/${c}_$tag.log 2>&1
    csv=$(grep -a 'SAFE_PYTEST: PROFILER CSV:' $L/${c}_$tag.log | tail -1 | sed 's/.*PROFILER CSV: //')
    [ -n "$csv" ] && break
  done
  ok=$(grep -ac '^BENCH_OK' $L/${c}_$tag.log)
  cp generated/profiler/.logs/profile_log_device.csv $L/${c}_$tag.device.csv
  python3 .claude/skills/tooling/scripts/ops_csv.py summary "$csv" --skip-first 1 > $L/${c}_$tag.summary.txt 2>&1
  spread=$(python3 $E/spans.py $L/${c}_$tag.device.csv 11 2>/dev/null | tail -1)
  echo "$c $tag ok=$ok $(sed -n 3p $L/${c}_$tag.summary.txt | awk '{print "median",$5,"min",$6,"max",$7,"BRISC",$11,"NCRISC",$12,"cores",$3}') | $spread"
done
