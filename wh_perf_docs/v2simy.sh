#!/bin/bash
# Versim on yyzeon09: top L1 movers, base vs a no-work change, without and with a packer resync every 32 pack calls
E='perf_eltwise_binary.py::test_perf_eltwise_binary[dest_acc:No-dest_sync:Half-unpack_to_dest:False-formats:Float16->Float16_b-math_op:Elwmul-math_fidelity:LoFi-transpose_srca:Yes-tile_dimensions:(32, 32)-input_dimensions:[256, 32]-acc_to_dest:False-run_types:[<PerfRunType.L1_TO_L1: 1>, <PerfRunType.UNPACK_ISOLATE: 2>, <PerfRunType.MATH_ISOLATE: 3>, <PerfRunType.PACK_ISOLATE: 4>, <PerfRunType.L1_CONGESTION: 5>]-loop_factor:32-is_perf:True]'
R='perf_reduce.py::test_perf_reduce[formats:Bfp8_b->Float16-dest_acc:No-reduce_dim:Scalar-pool_type:Sum]'
U='perf_unpack_a_bcast_eltwise.py::test_perf_col_tile_sdpa[formats:Float16_b->Float16_b-mathop:Elwadd-dest_acc:No-srca_reuse_count:4-math_fidelity:LoFi-input_dimensions:[64, 128]]'
C="LLK_SIM_BARRIER=1 LLK_PERF_INIT_LAUNCH=0"
L=LLK_PERF_RUN_TYPES=L1_CONGESTION; T=LLK_PERF_RUN_TYPES=L1_TO_L1
for r in 0 32; do
  P=REPRO_PACK_RESYNC=$r
  bash /tmp/vrun.sh ye_r${r}_b /tmp/v2 "$E" $C $L $P &
  bash /tmp/vrun.sh ye_r${r}_g /tmp/v2 "$E" $C $L $P LLK_RELEASE_GAP=1 &
  bash /tmp/vrun.sh yr_r${r}_b /tmp/v2 "$R" $C $L $P &
  bash /tmp/vrun.sh yr_r${r}_g /tmp/v2 "$R" $C $L $P LLK_RELEASE_GAP=1 &
  bash /tmp/vrun.sh yu_r${r}_b /tmp/v2 "$U" $C $T $P &
  bash /tmp/vrun.sh yu_r${r}_f /tmp/v2 "$U" $C $T $P LLK_FN_NOPS=1 &
done
wait; echo SIMYDONE > /tmp/v2simy.done
