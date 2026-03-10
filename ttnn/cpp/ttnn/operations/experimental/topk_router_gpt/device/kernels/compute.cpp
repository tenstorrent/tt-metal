// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute Kernel: Distributed Sender/Worker/Collector Architecture
//
// Sender: matmul K-slice × 1 N-tile → pack 1 partial tile
// Worker: matmul + add 2 sender partials (binary_dest_reuse) + add bias →
//         pack logit tile + index tile for collector
// Collector: continues from worker → merge 4 workers' logit tiles via
//            insertion-sort topk → softmax → pack final output

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/bcast.h"

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/reduce.h"

#include "api/compute/softmax.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t num_groups = get_named_compile_time_arg_val("num_groups");
    constexpr uint32_t topk_k = get_named_compile_time_arg_val("topk_k");

    // Runtime args
    uint32_t is_sender = get_arg_val<uint32_t>(0);
    uint32_t is_worker = get_arg_val<uint32_t>(1);
    uint32_t is_collector = get_arg_val<uint32_t>(2);
    uint32_t num_k_tiles = get_arg_val<uint32_t>(3);
    uint32_t n_tile_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t CB_WEIGHT = tt::CBIndex::c_0;
    constexpr uint32_t CB_INPUT = tt::CBIndex::c_1;
    constexpr uint32_t CB_PARTIAL_RECV = tt::CBIndex::c_2;
    constexpr uint32_t CB_LOCAL_OUT = tt::CBIndex::c_3;
    constexpr uint32_t CB_BIAS = tt::CBIndex::c_4;
    constexpr uint32_t CB_INDEX = tt::CBIndex::c_5;
    constexpr uint32_t CB_TOPK_VAL = tt::CBIndex::c_6;
    constexpr uint32_t CB_GATHERED_VAL = tt::CBIndex::c_8;
    constexpr uint32_t CB_GATHERED_IND = tt::CBIndex::c_9;
    constexpr uint32_t CB_INTERMED_VAL = tt::CBIndex::c_10;
    constexpr uint32_t CB_INTERMED_IND = tt::CBIndex::c_11;
    constexpr uint32_t CB_SOFTMAX_MASK = tt::CBIndex::c_12;
    constexpr uint32_t CB_SOFTMAX_TMP = tt::CBIndex::c_13;
    constexpr uint32_t CB_REDUCE_SCALAR = tt::CBIndex::c_14;
    constexpr uint32_t CB_BCAST_SCALER = tt::CBIndex::c_15;
    constexpr uint32_t CB_FINAL_OUT = tt::CBIndex::c_16;

    // =====================================================================
    // PHASE 1: Partial Matmul (all cores) — block-by-block
    // =====================================================================
    constexpr uint32_t BLOCK_SIZE = 2;

    // NOTE: dst_full_sync_en = false (half-sync mode). We use tile_regs_*
    // consistently throughout the kernel for correctness. acquire_dst/release_dst
    // must NOT be mixed with tile_regs_* in half-sync mode.
    mm_block_init(
        CB_INPUT,
        CB_WEIGHT,
        CB_LOCAL_OUT,
        /*transpose=*/0,
        /*ct_dim=*/1,
        /*rt_dim=*/1,
        /*kt_dim=*/1);
    tile_regs_acquire();

    uint32_t tiles_done = 0;
    while (tiles_done < num_k_tiles) {
        uint32_t block = num_k_tiles - tiles_done;
        if (block > BLOCK_SIZE) {
            block = BLOCK_SIZE;
        }

        cb_wait_front(CB_INPUT, block);
        cb_wait_front(CB_WEIGHT, block);

        for (uint32_t k = 0; k < block; k++) {
            matmul_block(
                CB_INPUT,
                CB_WEIGHT,
                /*in0_tile_index=*/k,
                /*in1_tile_index=*/k,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/1,
                /*rt_dim=*/1,
                /*kt_dim=*/1);
        }

        cb_pop_front(CB_INPUT, block);
        cb_pop_front(CB_WEIGHT, block);

        tiles_done += block;
    }

    if (is_sender) {
        // =================================================================
        // SENDER: pack partial and exit (DM1 sends it to worker)
        // =================================================================
        tile_regs_commit();
        cb_reserve_back(CB_LOCAL_OUT, 1);
        tile_regs_wait();
        pack_tile(0, CB_LOCAL_OUT);
        tile_regs_release();
        cb_push_back(CB_LOCAL_OUT, 1);
        return;
    }

    // =====================================================================
    // WORKER PATH: add sender partials + bias → pack logit tile
    // =====================================================================
    cb_wait_front(CB_PARTIAL_RECV, 2);

    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_PARTIAL_RECV);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_PARTIAL_RECV, 0, 0);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_PARTIAL_RECV, 1, 0);

    cb_pop_front(CB_PARTIAL_RECV, 2);

    // Add bias
    cb_wait_front(CB_BIAS, 1);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_BIAS, 0, 0);
    cb_pop_front(CB_BIAS, 1);

    // Pack complete logits to CB_TOPK_VAL
    tile_regs_commit();
    cb_reserve_back(CB_TOPK_VAL, 1);
    tile_regs_wait();
    pack_tile(0, CB_TOPK_VAL);
    tile_regs_release();
    cb_push_back(CB_TOPK_VAL, 1);

    if (!is_collector) {
        return;
    }

    // =====================================================================
    // COLLECTOR: 4-tile insertion-sort TopK on gathered logits
    // =====================================================================
    // NOTE: The topk hardware instruction always operates on 32-element vectors
    // within a tile, so k=32 and logk=5 are intrinsic tile-level constants,
    // NOT the user-facing topk_k. The actual user k is applied later during
    // output extraction (softmax mask in dm1.cpp, output packing).
    cb_wait_front(CB_GATHERED_VAL, num_groups);
    cb_wait_front(CB_GATHERED_IND, num_groups);

    ckernel::topk_tile_init();
    tile_regs_acquire();

    transpose_wh_init_short(CB_GATHERED_VAL);

    // Load tiles 0,1 (values → DST[0,1], indices → DST[2,3])
    transpose_wh_tile(CB_GATHERED_VAL, 0, 0);
    transpose_wh_tile(CB_GATHERED_VAL, 1, 1);
    transpose_wh_tile(CB_GATHERED_IND, 0, 2);
    transpose_wh_tile(CB_GATHERED_IND, 1, 3);

    // Sort first pair + merge → top-32 from 64 elements in DST[0,2]
    ckernel::topk_local_sort(0, /*idir=*/0, /*i_end_phase=*/4);
    ckernel::topk_merge(0, /*idir=*/0, /*k=*/32);

    // Insert tile 2
    transpose_wh_tile(CB_GATHERED_VAL, 2, 1);
    transpose_wh_tile(CB_GATHERED_IND, 2, 3);

    ckernel::topk_local_sort(0, /*idir=*/0, /*i_end_phase=*/4);
    ckernel::topk_merge(0, /*idir=*/0, /*k=*/32);

    // Insert tile 3
    transpose_wh_tile(CB_GATHERED_VAL, 3, 1);
    transpose_wh_tile(CB_GATHERED_IND, 3, 3);

    ckernel::topk_local_sort(0, /*idir=*/0, /*i_end_phase=*/4);
    ckernel::topk_merge(0, /*idir=*/0, /*k=*/32);

    // Rebuild final sorted order
    ckernel::topk_rebuild(0, /*idir=*/0, /*m_iter=*/0, /*k=*/32, /*logk=*/5, /*skip_second=*/true);

    tile_regs_commit();

    // Pack merged values → CB_INTERMED_VAL, indices → CB_INTERMED_IND
    cb_reserve_back(CB_INTERMED_VAL, 1);
    cb_reserve_back(CB_INTERMED_IND, 1);
    tile_regs_wait();
    pack_tile(0, CB_INTERMED_VAL);
    pack_tile(2, CB_INTERMED_IND);
    tile_regs_release();
    cb_push_back(CB_INTERMED_VAL, 1);
    cb_push_back(CB_INTERMED_IND, 1);

    cb_pop_front(CB_GATHERED_VAL, num_groups);
    cb_pop_front(CB_GATHERED_IND, num_groups);

    // --- Fused: transpose values+mask + transpose indices (one DST cycle) ---
    cb_wait_front(CB_INTERMED_VAL, 1);
    cb_wait_front(CB_INTERMED_IND, 1);
    cb_wait_front(CB_SOFTMAX_MASK, 1);
    cb_wait_front(CB_BCAST_SCALER, 1);

    tile_regs_acquire();
    transpose_wh_init_short(CB_INTERMED_VAL);
    transpose_wh_tile(CB_INTERMED_VAL, 0, 0);

    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_SOFTMAX_MASK);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_SOFTMAX_MASK, 0, 0);

    transpose_wh_init_short(CB_INTERMED_IND);
    transpose_wh_tile(CB_INTERMED_IND, 0, 1);
    tile_regs_commit();

    cb_pop_front(CB_INTERMED_VAL, 1);
    cb_pop_front(CB_INTERMED_IND, 1);
    cb_reserve_back(CB_SOFTMAX_TMP, 1);
    cb_reserve_back(CB_INTERMED_VAL, 1);

    tile_regs_wait();
    pack_tile(0, CB_SOFTMAX_TMP);
    pack_tile(1, CB_INTERMED_VAL);
    tile_regs_release();
    cb_push_back(CB_SOFTMAX_TMP, 1);
    cb_push_back(CB_INTERMED_VAL, 1);

    // =====================================================================
    // PHASE 4: Softmax on masked top-K values (collector only)
    // =====================================================================

    // Step 1: Find max per row
    cb_wait_front(CB_SOFTMAX_TMP, 1);
    cb_reserve_back(CB_REDUCE_SCALAR, 1);

    tile_regs_acquire();
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_SOFTMAX_TMP, CB_BCAST_SCALER, CB_REDUCE_SCALAR);
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_SOFTMAX_TMP, CB_BCAST_SCALER, 0, 0, 0);
    reduce_uninit(CB_REDUCE_SCALAR);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, CB_REDUCE_SCALAR);
    tile_regs_release();
    cb_push_back(CB_REDUCE_SCALAR, 1);

    // Step 2: Subtract max + Exp (fused)
    cb_wait_front(CB_REDUCE_SCALAR, 1);

    tile_regs_acquire();
    sub_bcast_cols_init_short(CB_SOFTMAX_TMP, CB_REDUCE_SCALAR);
    sub_tiles_bcast_cols(CB_SOFTMAX_TMP, CB_REDUCE_SCALAR, 0, 0, 0);
    exp_tile_init</*APPROX=*/1>();
    exp_tile</*APPROX=*/1>(0);
    tile_regs_commit();

    cb_pop_front(CB_SOFTMAX_TMP, 1);
    cb_reserve_back(CB_SOFTMAX_TMP, 1);
    tile_regs_wait();
    pack_tile(0, CB_SOFTMAX_TMP);
    tile_regs_release();
    cb_push_back(CB_SOFTMAX_TMP, 1);

    cb_pop_front(CB_REDUCE_SCALAR, 1);

    // Step 3: Reduce SUM per row + reciprocal
    cb_wait_front(CB_SOFTMAX_TMP, 1);
    cb_reserve_back(CB_REDUCE_SCALAR, 1);

    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SOFTMAX_TMP, CB_BCAST_SCALER, CB_REDUCE_SCALAR);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SOFTMAX_TMP, CB_BCAST_SCALER, 0, 0, 0);
    reduce_uninit(CB_REDUCE_SCALAR);
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, CB_REDUCE_SCALAR);
    tile_regs_release();
    cb_push_back(CB_REDUCE_SCALAR, 1);

    // Step 4: Multiply by 1/sum + copy indices (fused, one DST cycle)
    cb_wait_front(CB_SOFTMAX_TMP, 1);
    cb_wait_front(CB_REDUCE_SCALAR, 1);
    cb_wait_front(CB_INTERMED_VAL, 1);
    cb_reserve_back(CB_FINAL_OUT, 2);

    tile_regs_acquire();
    mul_bcast_cols_init_short(CB_SOFTMAX_TMP, CB_REDUCE_SCALAR);
    mul_tiles_bcast<BroadcastType::COL>(CB_SOFTMAX_TMP, CB_REDUCE_SCALAR, 0, 0, 0);

    copy_tile_to_dst_init_short(CB_INTERMED_VAL);
    copy_tile(CB_INTERMED_VAL, 0, 1);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, CB_FINAL_OUT);  // softmax weights
    pack_tile(1, CB_FINAL_OUT);  // indices
    tile_regs_release();

    cb_push_back(CB_FINAL_OUT, 2);
    cb_pop_front(CB_SOFTMAX_TMP, 1);
    cb_pop_front(CB_REDUCE_SCALAR, 1);
    cb_pop_front(CB_INTERMED_VAL, 1);

    cb_pop_front(CB_SOFTMAX_MASK, 1);
    cb_pop_front(CB_BCAST_SCALER, 1);
}
