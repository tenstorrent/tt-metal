#!/bin/bash
# Versim checks on /tmp/v2: WH-04 hold (config12601 PACK_ISOLATE) and M1c pads (config936 MATH_ISOLATE)
cp /tmp/tools/../vrun.sh /tmp/vrun.sh 2>/dev/null
Q=$(grep "HiFi2-matmul_config12601-5-1\]" /tmp/v2ids_perf_math_matmul.py.txt)
M=$(grep "LoFi-matmul_config936-0-1\]" /tmp/v2ids_perf_math_matmul.py.txt)
S="LLK_SIM_BARRIER=1"
bash /tmp/vrun.sh q0 /tmp/v2 "$Q" $S LLK_PERF_RUN_TYPES=PACK_ISOLATE &
bash /tmp/vrun.sh q1 /tmp/v2 "$Q" $S LLK_PERF_RUN_TYPES=PACK_ISOLATE LLK_FN_NOPS_THREADS=UNPACK,MATH LLK_THREAD_FN_NOPS=1 &
bash /tmp/vrun.sh q2 /tmp/v2 "$Q" $S LLK_PERF_RUN_TYPES=PACK_ISOLATE LLK_FN_NOPS_THREADS=UNPACK,MATH LLK_THREAD_FN_NOPS=1 LLK_NO_QUIET=1 &
bash /tmp/vrun.sh m0 /tmp/v2 "$M" $S LLK_PERF_RUN_TYPES=MATH_ISOLATE &
bash /tmp/vrun.sh m1 /tmp/v2 "$M" $S LLK_PERF_RUN_TYPES=MATH_ISOLATE LLK_FN_NOPS_THREADS=MATH LLK_THREAD_FN_NOPS=1 &
wait; echo V2SIMDONE > /tmp/v2sim.done
