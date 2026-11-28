#!/usr/bin/env bash
# run.sh <logdir> <case>:<cap bytes, 0 = the op's rule>[:<min cores per bank>] ...  (case bench, retried on a Tracy crash)
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
set +u; source python_env/bin/activate; set -u
export TT_METAL_HOME=$PWD
E=ttnn/ttnn/operations/g6_clone/perf_experiments/bank_cap_above_gate
# A bench file under ttnn/ re-imports ttnn under pytest: run a copy from the git-ignored generated/.
R=generated/meta/g6_clone/lanes/perf1_bank_cap; mkdir -p $R; cp $E/bench_cases.py $R/
export BANK_CAP_DIR=$PWD/$E
L=$1; shift; mkdir -p $L
for item in "$@"; do
  IFS=: read -r c b mn <<< "$item"; mn=${mn:-0}; tag=${c}_cap${b}_min${mn}
  for try in 1 2 3 4; do
    BANK_CAP_MIN=$mn BANK_CAP_BYTES=$b BENCH_CASES=$c TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile $R/bench_cases.py::test_bench > $L/$tag.log 2>&1
    csv=$(grep -a 'SAFE_PYTEST: PROFILER CSV:' $L/$tag.log | tail -1 | sed 's/.*PROFILER CSV: //')
    ok=$(grep -ac '^BENCH_OK' $L/$tag.log)
    [ -n "$csv" ] && [ "$ok" = 1 ] && break
  done
  python3 .claude/skills/tooling/scripts/ops_csv.py summary "$csv" --skip-first 1 > $L/$tag.summary.txt 2>&1
  row=$(grep -E '^ +11 ' $L/$tag.summary.txt | head -1)
  echo "$c cap=$b min=$mn ok=$ok $(echo $row | awk '{print "median",$5,"cores",$3}')"
done
