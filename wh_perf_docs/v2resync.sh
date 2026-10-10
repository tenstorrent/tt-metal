#!/bin/bash
# resync check on L1_TO_L1 and L1_CONGESTION: do the three no-work changes still move values with REPRO_PACK_RESYNC=N?
bash /tmp/v2setup.sh > /tmp/v2setup.out 2>&1
out=/tmp/v2c; mkdir -p $out; A=/proj_sw/user_dev/nstojic/v2r; mkdir -p $A
export CHIP_ARCH=wormhole USER=gitlab-ci LLK_PERF_RUN_TYPES=L1_TO_L1,L1_CONGESTION LLK_PERF_INIT_LAUNCH=0
run() { n=$1; shift; envs=("$@")
  [ -s $out/$n.log ] && grep -q "passed\|failed" $out/$n.log && return
  s=$(date +%s)
  ( cd /tmp/v2/tt_metal/tt-llk/tests/python_tests && source ../.venv/bin/activate && \
    env LLK_HOME=/tmp/v2/tt_metal/tt-llk RUNNER_TEMP=/tmp/build-v2r-$n NNG_SOCKET_NAME=v2r "${envs[@]}" python -u -m pytest -p no:randomly -q "${IDS[@]}" > $out/$n.log 2>&1 )
  for f in $(grep -o "Wrote run Parquet batch: [^ ]*" $out/$n.log | awk '{print $NF}' | sort -u); do mkdir -p $out/$n; cp $(dirname $f)/*/*.csv $out/$n/ 2>/dev/null; done
  rm -rf /tmp/build-v2r-$n; mkdir -p $A/$n; cp $out/$n.log $A/; cp $out/$n/*.csv $A/$n/ 2>/dev/null
  echo "$n $(tail -n1 $out/$n.log | tr -d '=') $(( $(date +%s) - s ))s" >> $out/progress_r.txt
}
mapfile -t IDS < /tmp/v2_set.txt
run b0
run b0_fn1  LLK_FN_NOPS=1
run b0_gap1 LLK_RELEASE_GAP=1
run b0_zr1  LLK_ZONE_RESERVE_NOPS=1
run r32     REPRO_PACK_RESYNC=32
run r32_fn1  REPRO_PACK_RESYNC=32 LLK_FN_NOPS=1
run r32_gap1 REPRO_PACK_RESYNC=32 LLK_RELEASE_GAP=1
run r32_zr1  REPRO_PACK_RESYNC=32 LLK_ZONE_RESERVE_NOPS=1
run r1      REPRO_PACK_RESYNC=1
run r1_fn1   REPRO_PACK_RESYNC=1 LLK_FN_NOPS=1
run r1_gap1  REPRO_PACK_RESYNC=1 LLK_RELEASE_GAP=1
run r1_zr1   REPRO_PACK_RESYNC=1 LLK_ZONE_RESERVE_NOPS=1
run b0_rerun X=2
echo RDONE >> $out/progress_r.txt
