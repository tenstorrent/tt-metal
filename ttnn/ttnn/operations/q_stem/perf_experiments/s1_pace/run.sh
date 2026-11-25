#!/bin/bash
# s1_pace: one ccl_perf run of a package copy.  run.sh <pkg> [knobs] [ablate] [outdir-suffix]
# e.g. run.sh q_stem_v "X_SPLIT_AHEAD=2" "SEND_PAYLOAD,S3,WKV_READ,OUT_WRITE" r2
cd "$(dirname "$0")/../../../../../.."
source python_env/bin/activate
S=ttnn/ttnn/operations/q_stem/perf_experiments/s1_pace
OUT=$S/results/${4:-r1}
mkdir -p $OUT
QSTEM_MODE=perf QSTEM_PKG=$1 QSTEM_KNOBS="$2" QSTEM_ABLATE="$3" CCL_PERF_OUT=$OUT timeout 3000 \
  scripts/run_safe_pytest.sh --no-precompile --import-mode=prepend $S/bench.py::test_bench -s > $OUT/last.log 2>&1
grep -E "SAFE_PYTEST_RESULT|passed|failed" $OUT/last.log | tail -2
tag=$(echo "$2" | sed 's/;/_/g; s/=//g; s/ //g; s/,/-/g; s/(//g; s/)//g')
abl=""; [ -n "$3" ] && abl="__abl_$(echo $3 | sed 's/,/_/g')"
f=$OUT/$1${tag:+__$tag}$abl.json
python -m eval.ccl_perf show $f | sed -n '1,3p;5p;7p'
