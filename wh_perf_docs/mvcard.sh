#!/bin/bash
# card legs on /tmp/mvt (his final tree + Versim switches) to find the exact configs of cases 2-6; results /tmp/mvc/<leg>/
out=/tmp/mvc; mkdir -p $out; A=/proj_sw/user_dev/nstojic/mvc; mkdir -p $A
cd /tmp/mvt/tt_metal/tt-llk/tests/python_tests && . ../.venv/bin/activate
export CHIP_ARCH=wormhole USER=gitlab-ci LLK_HOME=/tmp/mvt/tt_metal/tt-llk
leg() { # name "cflags" "pytest selection args..." [env...]
  n=$1; fl=$2; shift 2
  args=(); for a in "$@"; do if [ "${a:0:1}" = "@" ]; then mapfile -t -O ${#args[@]} args < "${a:1}"; else args+=("$a"); fi; done; set -- "${args[@]}"
  [ -s $out/$n.log ] && grep -q "passed\|failed" $out/$n.log && return
  s=$(date +%s)
  env RUNNER_TEMP=/tmp/build-mvc-$n LLK_EXP_CFLAGS="$fl" LLK_EXP_INIT_DUMP=$out/$n.init.jsonl $LEGENV python -u -m pytest -p no:randomly -q "$@" > $out/$n.log 2>&1
  for f in $(grep -o "Wrote run Parquet batch: [^ ]*" $out/$n.log | awk '{print $NF}' | sort -u); do mkdir -p $out/$n; cp $(dirname $f)/*/*.csv $out/$n/ 2>/dev/null; done
  mkdir -p $A/$n; cp $out/$n.log $out/$n.init.jsonl $A/ 2>/dev/null; cp $out/$n/*.csv $A/$n/ 2>/dev/null
  echo "$n $(tail -n1 $out/$n.log | tr -d =) $(( $(date +%s) - s ))s" >> $out/progress.txt
}
# cases 3-6: INIT
leg pdb_base "" perf_pack_dest_bank.py
leg pdb_u2 "-DLLK_EXP_NOP_UNPACK_INIT=2" perf_pack_dest_bank.py
leg pdb_m1 "-DLLK_EXP_MATH_NOP=1" perf_pack_dest_bank.py
leg pdb_rerun "" perf_pack_dest_bank.py
leg red_base "" @/tmp/ids_red.txt
leg red_m1 "-DLLK_EXP_MATH_NOP=1" @/tmp/ids_red.txt
leg utr_base "" perf_unpack_transpose.py
leg utr_u2 "-DLLK_EXP_NOP_UNPACK_INIT=2" perf_unpack_transpose.py
leg ebw_base "" @/tmp/ids_ebw.txt
leg ebw_wc "" @/tmp/ids_ebw.txt --enable-perf-counters
# case 2: scheduled release +1 cycle, L1_TO_L1
LEGENV="LLK_PERF_RUN_TYPES=L1_TO_L1"
leg tdf_g0 "" perf_transpose_dest.py perf_fast_tilize.py
leg tdf_g1 "-DLLK_RELEASE_GAP_SCHED=1" perf_transpose_dest.py perf_fast_tilize.py
echo MVCDONE >> $out/progress.txt
