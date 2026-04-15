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

    // Run-time arguments (shared layout with dm0 and dm1)
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto weight_addr = get_arg_val<uint32_t>(argidx++);
    const auto input_addr = get_arg_val<uint32_t>(argidx++);
    const auto bias_addr = get_arg_val<uint32_t>(argidx++);
    const auto sem_partial_ready = get_arg_val<uint32_t>(argidx++);
    const auto is_sender = get_arg_val<uint32_t>(argidx++);
    const auto is_worker = get_arg_val<uint32_t>(argidx++);
    const auto is_collector = get_arg_val<uint32_t>(argidx++);
    const auto num_k_tiles = get_arg_val<uint32_t>(argidx++);
    const auto k_tile_offset = get_arg_val<uint32_t>(argidx++);
    const auto n_tile_id = get_arg_val<uint32_t>(argidx++);
    const auto worker_phys_x = get_arg_val<uint32_t>(argidx++);
    const auto worker_phys_y = get_arg_val<uint32_t>(argidx++);
    const auto sender_slot = get_arg_val<uint32_t>(argidx++);
    const auto worker_gather_slot = get_arg_val<uint32_t>(argidx++);
    const auto sem_topk_ready = get_arg_val<uint32_t>(argidx++);
    const auto indices_rm_addr = get_arg_val<uint32_t>(argidx++);
    const auto weights_rm_addr = get_arg_val<uint32_t>(argidx++);
    const auto aligned_page_size = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_weight = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_partial_recv = tt::CBIndex::c_2;
    constexpr auto cb_local_out = tt::CBIndex::c_3;
    constexpr auto cb_bias = tt::CBIndex::c_4;
    constexpr auto cb_index = tt::CBIndex::c_5;
    constexpr auto cb_topk_val = tt::CBIndex::c_6;
    constexpr auto cb_gathered_val = tt::CBIndex::c_8;
    constexpr auto cb_gathered_ind = tt::CBIndex::c_9;
    constexpr auto cb_intermed_val = tt::CBIndex::c_10;
    constexpr auto cb_intermed_ind = tt::CBIndex::c_11;
    constexpr auto cb_softmax_mask = tt::CBIndex::c_12;
    constexpr auto cb_softmax_tmp = tt::CBIndex::c_13;
    constexpr auto cb_reduce_scalar = tt::CBIndex::c_14;
    constexpr auto cb_bcast_scaler = tt::CBIndex::c_15;
    constexpr auto cb_final_out = tt::CBIndex::c_16;

    // =====================================================================
    // PHASE 1: Partial Matmul (all cores) — block-by-block
    // =====================================================================
    constexpr uint32_t BLOCK_SIZE = 2;

    // NOTE: dst_full_sync_en = false (half-sync mode). We use tile_regs_*
    // consistently throughout the kernel for correctness. acquire_dst/release_dst
    // must NOT be mixed with tile_regs_* in half-sync mode.
    mm_block_init(
        cb_input,
        cb_weight,
        cb_local_out,
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

        cb_wait_front(cb_input, block);
        cb_wait_front(cb_weight, block);

        for (uint32_t k = 0; k < block; k++) {
            matmul_block(
                cb_input,
                cb_weight,
                /*in0_tile_index=*/k,
                /*in1_tile_index=*/k,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/1,
                /*rt_dim=*/1,
                /*kt_dim=*/1);
        }

        cb_pop_front(cb_input, block);
        cb_pop_front(cb_weight, block);

        tiles_done += block;
    }

    if (is_sender) {
        // =================================================================
        // SENDER: pack partial and exit (DM1 sends it to worker)
        // =================================================================
        tile_regs_commit();
        cb_reserve_back(cb_local_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_local_out);
        tile_regs_release();
        cb_push_back(cb_local_out, 1);
        return;
    }

    // =====================================================================
    // WORKER PATH: add sender partials + bias → pack logit tile
    // =====================================================================
    cb_wait_front(cb_partial_recv, 2);

    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_partial_recv);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_partial_recv, 0, 0);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_partial_recv, 1, 0);

    cb_pop_front(cb_partial_recv, 2);

    // Add bias
    cb_wait_front(cb_bias, 1);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_bias, 0, 0);
    cb_pop_front(cb_bias, 1);

    // Pack complete logits to cb_topk_val
    tile_regs_commit();
    cb_reserve_back(cb_topk_val, 1);
    tile_regs_wait();
    pack_tile(0, cb_topk_val);
    tile_regs_release();
    cb_push_back(cb_topk_val, 1);

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
    cb_wait_front(cb_gathered_val, num_groups);
    cb_wait_front(cb_gathered_ind, num_groups);

    ckernel::topk_tile_init();
    tile_regs_acquire();

    transpose_wh_init_short(cb_gathered_val);

    // Load tiles 0,1 (values → DST[0,1], indices → DST[2,3])
    transpose_wh_tile(cb_gathered_val, 0, 0);
    transpose_wh_tile(cb_gathered_val, 1, 1);
    transpose_wh_tile(cb_gathered_ind, 0, 2);
    transpose_wh_tile(cb_gathered_ind, 1, 3);

    // Sort first pair + merge → top-32 from 64 elements in DST[0,2]
    ckernel::topk_local_sort(0, /*idir=*/0, /*i_end_phase=*/4);
    ckernel::topk_merge(0, /*idir=*/0, /*k=*/32);

    // Insert tile 2
    transpose_wh_tile(cb_gathered_val, 2, 1);
    transpose_wh_tile(cb_gathered_ind, 2, 3);

    ckernel::topk_local_sort(0, /*idir=*/0, /*i_end_phase=*/4);
    ckernel::topk_merge(0, /*idir=*/0, /*k=*/32);

    // Insert tile 3
    transpose_wh_tile(cb_gathered_val, 3, 1);
    transpose_wh_tile(cb_gathered_ind, 3, 3);

    ckernel::topk_local_sort(0, /*idir=*/0, /*i_end_phase=*/4);
    ckernel::topk_merge(0, /*idir=*/0, /*k=*/32);

    // Rebuild final sorted order
    ckernel::topk_rebuild(0, /*idir=*/0, /*m_iter=*/0, /*k=*/32, /*logk=*/5, /*skip_second=*/true);

    tile_regs_commit();

    // Pack merged values → cb_intermed_val, indices → cb_intermed_ind
    cb_reserve_back(cb_intermed_val, 1);
    cb_reserve_back(cb_intermed_ind, 1);
    tile_regs_wait();
    pack_tile(0, cb_intermed_val);
    pack_tile(2, cb_intermed_ind);
    tile_regs_release();
    cb_push_back(cb_intermed_val, 1);
    cb_push_back(cb_intermed_ind, 1);

    cb_pop_front(cb_gathered_val, num_groups);
    cb_pop_front(cb_gathered_ind, num_groups);

    // Fused: transpose values+mask + transpose indices (one DST cycle)
    cb_wait_front(cb_intermed_val, 1);
    cb_wait_front(cb_intermed_ind, 1);
    cb_wait_front(cb_softmax_mask, 1);
    cb_wait_front(cb_bcast_scaler, 1);

    tile_regs_acquire();
    transpose_wh_init_short(cb_intermed_val);
    transpose_wh_tile(cb_intermed_val, 0, 0);

    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_softmax_mask);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_softmax_mask, 0, 0);

    transpose_wh_init_short(cb_intermed_ind);
    transpose_wh_tile(cb_intermed_ind, 0, 1);
    tile_regs_commit();

    cb_pop_front(cb_intermed_val, 1);
    cb_pop_front(cb_intermed_ind, 1);
    cb_reserve_back(cb_softmax_tmp, 1);
    cb_reserve_back(cb_intermed_val, 1);

    tile_regs_wait();
    pack_tile(0, cb_softmax_tmp);
    pack_tile(1, cb_intermed_val);
    tile_regs_release();
    cb_push_back(cb_softmax_tmp, 1);
    cb_push_back(cb_intermed_val, 1);

    // =====================================================================
    // PHASE 4: Softmax on masked top-K values (collector only)
    // =====================================================================

    // Step 1: Find max per row
    cb_wait_front(cb_softmax_tmp, 1);
    cb_reserve_back(cb_reduce_scalar, 1);

    tile_regs_acquire();
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_softmax_tmp, cb_bcast_scaler, cb_reduce_scalar);
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_softmax_tmp, cb_bcast_scaler, 0, 0, 0);
    reduce_uninit(cb_reduce_scalar);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_reduce_scalar);
    tile_regs_release();
    cb_push_back(cb_reduce_scalar, 1);

    // Step 2: Subtract max + Exp (fused)
    cb_wait_front(cb_reduce_scalar, 1);

    tile_regs_acquire();
    sub_bcast_cols_init_short(cb_softmax_tmp, cb_reduce_scalar);
    sub_tiles_bcast_cols(cb_softmax_tmp, cb_reduce_scalar, 0, 0, 0);
    exp_tile_init</*APPROX=*/1>();
    exp_tile</*APPROX=*/1>(0);
    tile_regs_commit();

    cb_pop_front(cb_softmax_tmp, 1);
    cb_reserve_back(cb_softmax_tmp, 1);
    tile_regs_wait();
    pack_tile(0, cb_softmax_tmp);
    tile_regs_release();
    cb_push_back(cb_softmax_tmp, 1);

    cb_pop_front(cb_reduce_scalar, 1);

    // Step 3: Reduce SUM per row + reciprocal
    cb_wait_front(cb_softmax_tmp, 1);
    cb_reserve_back(cb_reduce_scalar, 1);

    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, true>(cb_softmax_tmp, cb_bcast_scaler, cb_reduce_scalar);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_softmax_tmp, cb_bcast_scaler, 0, 0, 0);
    reduce_uninit<true>(cb_reduce_scalar);
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_reduce_scalar);
    tile_regs_release();
    cb_push_back(cb_reduce_scalar, 1);

    // Step 4: Multiply by 1/sum + copy indices (fused, one DST cycle)
    cb_wait_front(cb_softmax_tmp, 1);
    cb_wait_front(cb_reduce_scalar, 1);
    cb_wait_front(cb_intermed_val, 1);
    cb_reserve_back(cb_final_out, 2);

    tile_regs_acquire();
    mul_bcast_cols_init_short(cb_softmax_tmp, cb_reduce_scalar);
    mul_tiles_bcast<BroadcastType::COL>(cb_softmax_tmp, cb_reduce_scalar, 0, 0, 0);

    copy_tile_to_dst_init_short(cb_intermed_val);
    copy_tile(cb_intermed_val, 0, 1);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_final_out);  // softmax weights
    pack_tile(1, cb_final_out);  // indices
    tile_regs_release();

    cb_push_back(cb_final_out, 2);
    cb_pop_front(cb_softmax_tmp, 1);
    cb_pop_front(cb_reduce_scalar, 1);
    cb_pop_front(cb_intermed_val, 1);

    cb_pop_front(cb_softmax_mask, 1);
    cb_pop_front(cb_bcast_scaler, 1);
}
