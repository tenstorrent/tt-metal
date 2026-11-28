#!/usr/bin/env bash
# Issue-path lane runs: run.sh <logdir> <case>:<variant|op> ...; one bench process per item, in the given order.
# variant "op" = the op's own kernels; anything else = kernels_<variant>/ in this folder (same CT/RT layout,
# host unchanged). Each item prints the ops_csv.py summary row (--skip-first 1: median of the 11 launches
# after the first). IMPL=stock in the environment times stock ttnn.clone instead (variant ignored).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
set +u; source python_env/bin/activate; set -u
export TT_METAL_HOME=$PWD
E=ttnn/ttnn/operations/g6_clone/perf_experiments/issue_path
# A bench file under ttnn/ re-imports ttnn under pytest: run a copy from the git-ignored generated/.
R=generated/meta/g6_clone/lanes/perf1_issue_path; mkdir -p $R; cp $E/bench_cases.py $R/
export ISSUE_KERNEL_ROOT=$PWD/$E
L=$1; shift; mkdir -p $L
for item in "$@"; do
  IFS=: read -r c v <<< "$item"; [ "$v" = op ] && v=""
  impl=${IMPL:-new}
  tag=${c}_${v:-op}_${impl}
  csv=""
  for try in 1 2 3; do
    ISSUE_VARIANT=$v BENCH_IMPL=$impl BENCH_CASES=$c TRACY_NO_WEB_SERVER=1 \
      scripts/run_safe_pytest.sh --profile --no-py-zones $R/bench_cases.py::test_bench > $L/$tag.log 2>&1
    csv=$(grep -a 'SAFE_PYTEST: PROFILER CSV:' $L/$tag.log | tail -1 | sed 's/.*PROFILER CSV: //')
    [ -n "$csv" ] && grep -aq '^BENCH_OK' $L/$tag.log || continue
    python3 .claude/skills/tooling/scripts/ops_csv.py summary "$csv" --skip-first 1 > $L/$tag.summary.txt 2>&1
    # A capture that matched fewer than the 11 measured launches is retried (Tracy host/device mismatch).
    n=$(awk '/^ +[0-9]+ +[0-9]+ /{print $1; exit}' $L/$tag.summary.txt)
    [ "$n" = 11 ] && break
  done
  cp generated/profiler/.logs/profile_log_device.csv $L/$tag.device.csv 2>/dev/null
  ok=$(grep -ac '^BENCH_OK' $L/$tag.log)
  kd=$(grep -a '^BENCH_KERNEL_DIR' $L/$tag.log | head -1 | sed 's/.*dir=//')
  row=$(awk '/^ +[0-9]+ +[0-9]+ /{print; exit}' $L/$tag.summary.txt)
  echo "$c ${v:-op} $impl ok=$ok kdir=$(basename "$kd") :: $row"
done
