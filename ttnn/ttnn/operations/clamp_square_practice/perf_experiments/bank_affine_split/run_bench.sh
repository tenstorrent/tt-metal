#!/bin/bash
# usage: run_bench.sh <shapes csv> <variants csv> <tag>   (profiled, one launch per (variant, shape))
cd /localdev/jmitrovic/tt-metal-codegen
E=ttnn/ttnn/operations/clamp_square_practice/perf_experiments/bank_affine_split
T=tests/ttnn/unit_tests/operations/clamp_square_practice/test_perfexp_bank_affine_split.py
BAS_SHAPES=$1 BAS_VARIANTS=$2 scripts/run_safe_pytest.sh --profile --run-all $T::test_perf > $E/prof_$3.log 2>&1
grep -E "passed|failed" $E/prof_$3.log | tail -1
R=$(ls -td /localdev/jmitrovic/tt-metal/generated/profiler/reports/*/ | head -1)
L=$(grep -oE "BENCH_ROW variant=[a-z_]+ shape=\([0-9, ]+\)" $E/prof_$3.log | sed -E 's/BENCH_ROW variant=//; s/ shape=\(/@/; s/\)//; s/, /x/g' | paste -sd,)
echo "report $R"
python $E/parse.py $R $L | cut -c1-120
