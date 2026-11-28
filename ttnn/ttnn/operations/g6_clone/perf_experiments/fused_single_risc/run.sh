#!/usr/bin/env bash
# fused_single_risc bench runs: run.sh <logdir> <case>:<variant|base|stock> ...; one bench process per item,
# in the pipeline form (case bench, 12 launches, median of the 11 after the first).
set -uo pipefail
cd /localdev/pmilojevic/campaign/bh-lb-18/slots/chip0
set +u; source python_env/bin/activate; set -u
export TT_METAL_HOME=$PWD
E=ttnn/ttnn/operations/g6_clone/perf_experiments/fused_single_risc
# A bench file under ttnn/ re-imports ttnn under pytest: run a copy from the git-ignored generated/.
R=generated/meta/g6_clone/lanes/perf1_fused_single_risc; mkdir -p $R; cp $E/bench_cases.py $E/fused_plugin.py $R/
L=$1; shift; mkdir -p $L
for item in "$@"; do
  IFS=: read -r c v <<< "$item"
  impl=new; fv=$v; op=GenericOp
  [ "$v" = base ] && fv=""
  [ "$v" = stock ] && { fv=""; impl=stock; op=CloneOperation; }
  tag=${c}_${v}
  for try in 1 2 3; do
    FUSED_VARIANT=$fv BENCH_IMPL=$impl BENCH_CASES=$c TRACY_NO_WEB_SERVER=1 scripts/run_safe_pytest.sh --profile --no-py-zones $R/bench_cases.py::test_bench > $L/$tag.log 2>&1
    csv=$(grep -a 'SAFE_PYTEST: PROFILER CSV:' $L/$tag.log | tail -1 | sed 's/.*PROFILER CSV: //')
    ok=$(grep -ac '^BENCH_OK' $L/$tag.log)
    [ -n "$csv" ] && [ "$ok" = 1 ] && break
  done
  python3 .claude/skills/tooling/scripts/ops_csv.py summary "$csv" --skip-first 1 > $L/$tag.summary.txt 2>&1
  echo "$tag ok=$ok $(grep -a '^BENCH_CASE' $L/$tag.log | grep -o 'fused=[a-z_0-9]*') $(grep -a -A1 "launches " $L/$tag.summary.txt | tail -1 | awk '{print "median",$5,"min",$6,"max",$7,"cores",$3,"BRISC",$11,"NCRISC",$12}')"
done
