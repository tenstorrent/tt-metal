#!/bin/bash
# Versim (vrun3: VCD through a FIFO, own versim with a Verilator error limit) on /tmp/mvt: INIT cases 3 and 4, release case 2
P8='perf_pack_dest_bank.py::test_perf_pack_dest_bank[formats:Float16_b->Float16_b-dest_acc:No-l1_acc:No-num_faces:4-tilize:No-dest_index:0-num_blocks:1-num_tiles_in_block:8-loop_factor:8]'
P4='perf_pack_dest_bank.py::test_perf_pack_dest_bank[formats:Float16_b->Float16_b-dest_acc:No-l1_acc:No-num_faces:4-tilize:No-dest_index:0-num_blocks:1-num_tiles_in_block:4-loop_factor:8]'
R='perf_reduce.py::test_perf_reduce[formats:Float32->Bfp8_b-dest_acc:No-reduce_dim:Row-pool_type:Average]'
T='perf_transpose_dest.py::test_perf_transpose_dest[formats:Float16_b->Float16_b-unpack_transpose_faces:No-math_transpose_faces:Yes]'
C="LLK_SIM_INIT=1"
run() { rm -rf /proj_sw/user_dev/nstojic/versim-runs/$1 /tmp/v_$1.done; bash /tmp/vrun3.sh "$@"; }
run vp4_b 5731 "$P4" $C &
run vp4_u 5732 "$P4" $C LLK_EXP_CFLAGS=-DLLK_EXP_NOP_UNPACK_INIT=2 &
run vp8_b 5733 "$P8" $C &
run vp8_u 5734 "$P8" $C LLK_EXP_CFLAGS=-DLLK_EXP_NOP_UNPACK_INIT=2 &
run vp8_m 5735 "$P8" $C LLK_EXP_CFLAGS=-DLLK_EXP_MATH_NOP=1 &
run vr_b 5736 "$R" $C LLK_PERF_RUN_TYPES=MATH_ISOLATE &
run vr_m 5737 "$R" $C LLK_PERF_RUN_TYPES=MATH_ISOLATE LLK_EXP_CFLAGS=-DLLK_EXP_MATH_NOP=1 &
run vt_g0 5738 "$T" LLK_SIM_WARMUP=1 LLK_PERF_RUN_TYPES=L1_TO_L1 &
run vt_g1 5739 "$T" LLK_SIM_WARMUP=1 LLK_PERF_RUN_TYPES=L1_TO_L1 LLK_EXP_CFLAGS=-DLLK_RELEASE_GAP_SCHED=1 &
wait; echo VINIT2DONE > /tmp/vinit2.done
