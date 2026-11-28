#!/usr/bin/env bash
# Step-1 ablation runs: run.sh <logdir> <case>:<variant|full> ...; one bench process per item.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
set +u; source python_env/bin/activate; set -u
export TT_METAL_HOME=$PWD
E=ttnn/ttnn/operations/g6_clone/perf_experiments/step1_ablation
# A bench file under ttnn/ re-imports ttnn under pytest: run a copy from the git-ignored generated/.
R=generated/meta/g6_clone/lanes/perf1_step1_ablation; mkdir -p $R; cp $E/bench_cases.py $R/
export ABL_KERNEL_ROOT=$PWD/$E
L=$1; shift; mkdir -p $L
for item in "$@"; do
  IFS=: read -r c v <<< "$item"; [ "$v" = full ] && v=""
  tag=${c}_${v:-full}
  for try in 1 2 3; do
    ABL_VARIANT=$v BENCH_CASES=$c TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile $R/bench_cases.py::test_bench > $L/$tag.log 2>&1
    csv=$(grep -a 'SAFE_PYTEST: PROFILER CSV:' $L/$tag.log | tail -1 | sed 's/.*PROFILER CSV: //')
    [ -n "$csv" ] && break
  done
  cp generated/profiler/.logs/profile_log_device.csv $L/$tag.device.csv
  python3 .claude/skills/tooling/scripts/ops_csv.py summary "$csv" --skip-first 1 > $L/$tag.summary.txt 2>&1
  echo "$tag $(sed -n 3p $L/$tag.summary.txt | awk '{print "median",$5,"BRISC",$11,"NCRISC",$12,"cores",$3}')"
done
