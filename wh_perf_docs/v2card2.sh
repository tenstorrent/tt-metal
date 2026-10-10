#!/bin/bash
# second card queue (time budget): waits for base, then the PACK_ISOLATE only checks, then full set checks
out=/tmp/v2c; A=/proj_sw/user_dev/nstojic/v2c; mkdir -p $A
while kill -0 95617 2>/dev/null; do sleep 20; done
n=base; for f in $(grep -o "Wrote run Parquet batch: [^ ]*" $out/$n.log | awk '{print $NF}' | sort -u); do mkdir -p $out/$n; cp $(dirname $f)/*/*.csv $out/$n/ 2>/dev/null; done
rm -rf /tmp/build-v2c-$n; mkdir -p $A/$n; cp $out/$n.log $A/; cp $out/$n/*.csv $A/$n/ 2>/dev/null; echo "$n $(tail -n1 $out/$n.log)" >> $out/progress.txt
export CHIP_ARCH=wormhole USER=gitlab-ci
run() { n=$1; shift; envs=("$@")
  [ -s $out/$n.log ] && grep -q "passed\|failed" $out/$n.log && { echo "$n done before"; return; }
  s=$(date +%s)
  ( cd /tmp/v2/tt_metal/tt-llk/tests/python_tests && source ../.venv/bin/activate && \
    env LLK_HOME=/tmp/v2/tt_metal/tt-llk RUNNER_TEMP=/tmp/build-v2c-$n NNG_SOCKET_NAME=v2c "${envs[@]}" python -u -m pytest -p no:randomly -q "${IDS[@]}" > $out/$n.log 2>&1 )
  for f in $(grep -o "Wrote run Parquet batch: [^ ]*" $out/$n.log | awk '{print $NF}' | sort -u); do mkdir -p $out/$n; cp $(dirname $f)/*/*.csv $out/$n/ 2>/dev/null; done
  rm -rf /tmp/build-v2c-$n; mkdir -p $A/$n; cp $out/$n.log $A/; cp $out/$n/*.csv $A/$n/ 2>/dev/null
  echo "$n $(tail -n1 $out/$n.log) $(( $(date +%s) - s ))s" >> $out/progress.txt
}
mapfile -t IDS < /tmp/v2_set.txt
P=LLK_PERF_RUN_TYPES=PACK_ISOLATE
run pi_um1     $P LLK_FN_NOPS_THREADS=UNPACK,MATH LLK_THREAD_FN_NOPS=1
run pi_noq_um1 $P LLK_NO_QUIET=1 LLK_FN_NOPS_THREADS=UNPACK,MATH LLK_THREAD_FN_NOPS=1
run pi_noq     $P LLK_NO_QUIET=1
run bn1   LLK_BRISC_FN_NOPS=1
run gap1  LLK_RELEASE_GAP=1
run fn1   LLK_FN_NOPS=1
run base2 X=2
run pi_pk1     $P LLK_FN_NOPS_THREADS=PACK LLK_THREAD_FN_NOPS=1
echo V2CDONE >> $out/progress.txt
