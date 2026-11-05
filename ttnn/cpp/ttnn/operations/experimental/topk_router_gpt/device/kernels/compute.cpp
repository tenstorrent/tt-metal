// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute Kernel: Distributed Sender/Worker/Collector Architecture
//
// Sender: matmul K-slice × 1 N-tile → pack 1 partial tile
// Worker: matmul + add sender partials (binary_dest_reuse) + add bias →
//         pack logit tile + index tile for collector
// Collector: continues from worker → merge 4 workers' logit tiles via
//            insertion-sort topk → softmax → pack final output

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/topk.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/reduce.h"

#include "api/compute/softmax.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t num_groups = get_arg(args::num_groups);
    constexpr uint32_t num_senders = get_arg(args::num_senders);
    constexpr uint32_t topk_k = get_arg(args::topk_k);

    // Run-time arguments (shared layout with dm0 and dm1)
    const auto dram_bank_id = get_arg(args::dram_bank_id);
    const auto vchannel = get_arg(args::vchannel);
    const auto is_sender = get_arg(args::is_sender);
    const auto is_worker = get_arg(args::is_worker);
    const auto is_collector = get_arg(args::is_collector);
    const auto num_k_tiles = get_arg(args::num_k_tiles);
    const auto k_tile_offset = get_arg(args::k_tile_offset);
    const auto n_tile_id = get_arg(args::n_tile_id);
    const auto worker_phys_x = get_arg(args::worker_phys_x);
    const auto worker_phys_y = get_arg(args::worker_phys_y);
    const auto sender_slot = get_arg(args::sender_slot);
    const auto worker_gather_slot = get_arg(args::worker_gather_slot);

    // DFBs
    DataflowBuffer dfb_weight(dfb::weight);
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_partial_recv(dfb::partial_recv);
    DataflowBuffer dfb_local_out(dfb::local_out);
    DataflowBuffer dfb_bias(dfb::bias);
    DataflowBuffer dfb_topk_val(dfb::topk_val);
    DataflowBuffer dfb_gathered_val(dfb::gathered_val);
    DataflowBuffer dfb_gathered_ind(dfb::gathered_ind);
    DataflowBuffer dfb_intermed_val(dfb::intermed_val);
    DataflowBuffer dfb_intermed_ind(dfb::intermed_ind);
    DataflowBuffer dfb_softmax_mask(dfb::softmax_mask);
    DataflowBuffer dfb_softmax_tmp(dfb::softmax_tmp);
    DataflowBuffer dfb_reduce_scalar(dfb::reduce_scalar);
    DataflowBuffer dfb_bcast_scaler(dfb::bcast_scaler);
    DataflowBuffer dfb_final_out(dfb::final_out);

    // =====================================================================
    // PHASE 1: Partial Matmul (all cores) — block-by-block
    // =====================================================================
    constexpr uint32_t BLOCK_SIZE = 2;

    // NOTE: dst_full_sync_en = false (half-sync mode). We use tile_regs_*
    // consistently throughout the kernel for correctness. acquire_dst/release_dst
    // must NOT be mixed with tile_regs_* in half-sync mode.
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::input, dfb::weight, dfb::local_out);
    matmul_block_init(dfb::input, dfb::weight, /*transpose=*/0, /*ct_dim=*/1, /*rt_dim=*/1, /*kt_dim=*/1);
    tile_regs_acquire();

    uint32_t tiles_done = 0;
    while (tiles_done < num_k_tiles) {
        uint32_t block = num_k_tiles - tiles_done;
        if (block > BLOCK_SIZE) {
            block = BLOCK_SIZE;
        }

        dfb_input.wait_front(block);
        dfb_weight.wait_front(block);

        for (uint32_t k = 0; k < block; k++) {
            matmul_block(
                dfb::input,
                dfb::weight,
                /*in0_tile_index=*/k,
                /*in1_tile_index=*/k,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/1,
                /*rt_dim=*/1,
                /*kt_dim=*/1);
        }

        dfb_input.pop_front(block);
        dfb_weight.pop_front(block);

        tiles_done += block;
    }

    if (is_sender) {
        // =================================================================
        // SENDER: pack partial and exit (DM1 sends it to worker)
        // =================================================================
        tile_regs_commit();
        dfb_local_out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::local_out);
        tile_regs_release();
        dfb_local_out.push_back(1);
        return;
    }

    // =====================================================================
    // WORKER PATH: add sender partials + bias → pack logit tile
    // =====================================================================
    dfb_partial_recv.wait_front(num_senders);

    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::partial_recv);
    for (uint32_t sender = 0; sender < num_senders; sender++) {
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::partial_recv, sender, 0);
    }

    dfb_partial_recv.pop_front(num_senders);

    // Add bias
    dfb_bias.wait_front(1);
    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::bias, 0, 0);
    dfb_bias.pop_front(1);

    // Pack complete logits to dfb::topk_val
    tile_regs_commit();
    dfb_topk_val.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, dfb::topk_val);
    tile_regs_release();
    dfb_topk_val.push_back(1);

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
    dfb_gathered_val.wait_front(num_groups);
    dfb_gathered_ind.wait_front(num_groups);

    ckernel::topk_tile_init();
    tile_regs_acquire();

    transpose_init(dfb::gathered_val);

    // Load tiles 0,1 (values → DST[0,1], indices → DST[2,3])
    transpose_tile(dfb::gathered_val, 0, 0);
    transpose_tile(dfb::gathered_val, 1, 1);
    transpose_tile(dfb::gathered_ind, 0, 2);
    transpose_tile(dfb::gathered_ind, 1, 3);

    // Sort first pair + merge → top-32 from 64 elements in DST[0,2]
    ckernel::topk_local_sort(0, /*idir=*/0, /*i_end_phase=*/4);
    ckernel::topk_merge(0, /*idir=*/0, /*k=*/32);

    // Insert tile 2
    transpose_tile(dfb::gathered_val, 2, 1);
    transpose_tile(dfb::gathered_ind, 2, 3);

    ckernel::topk_local_sort(0, /*idir=*/0, /*i_end_phase=*/4);
    ckernel::topk_merge(0, /*idir=*/0, /*k=*/32);

    // Insert tile 3
    transpose_tile(dfb::gathered_val, 3, 1);
    transpose_tile(dfb::gathered_ind, 3, 3);

    ckernel::topk_local_sort(0, /*idir=*/0, /*i_end_phase=*/4);
    ckernel::topk_merge(0, /*idir=*/0, /*k=*/32);

    // Rebuild final sorted order
    ckernel::topk_rebuild(0, /*idir=*/0, /*m_iter=*/0, /*k=*/32, /*logk=*/5, /*skip_second=*/true);

    tile_regs_commit();

    // Pack merged values → dfb::intermed_val, indices → dfb::intermed_ind
    dfb_intermed_val.reserve_back(1);
    dfb_intermed_ind.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, dfb::intermed_val);
    pack_tile(2, dfb::intermed_ind);
    tile_regs_release();
    dfb_intermed_val.push_back(1);
    dfb_intermed_ind.push_back(1);

    dfb_gathered_val.pop_front(num_groups);
    dfb_gathered_ind.pop_front(num_groups);

    // Fused: transpose values+mask + transpose indices (one DST cycle)
    dfb_intermed_val.wait_front(1);
    dfb_intermed_ind.wait_front(1);
    dfb_softmax_mask.wait_front(1);
    dfb_bcast_scaler.wait_front(1);

    tile_regs_acquire();
    transpose_init(dfb::intermed_val);
    transpose_tile(dfb::intermed_val, 0, 0);

    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::softmax_mask);
    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb::softmax_mask, 0, 0);

    transpose_init(dfb::intermed_ind);
    transpose_tile(dfb::intermed_ind, 0, 1);
    tile_regs_commit();

    dfb_intermed_val.pop_front(1);
    dfb_intermed_ind.pop_front(1);
    dfb_softmax_tmp.reserve_back(1);
    dfb_intermed_val.reserve_back(1);

    tile_regs_wait();
    pack_tile(0, dfb::softmax_tmp);
    pack_tile(1, dfb::intermed_val);
    tile_regs_release();
    dfb_softmax_tmp.push_back(1);
    dfb_intermed_val.push_back(1);

    // =====================================================================
    // PHASE 4: Softmax on masked top-K values (collector only)
    // =====================================================================

    // Step 1: Find max per row
    dfb_softmax_tmp.wait_front(1);
    dfb_reduce_scalar.reserve_back(1);

    tile_regs_acquire();
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::softmax_tmp, dfb::bcast_scaler, dfb::reduce_scalar);
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb::softmax_tmp, dfb::bcast_scaler, 0, 0, 0);
    reduce_uninit(dfb::reduce_scalar);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, dfb::reduce_scalar);
    tile_regs_release();
    dfb_reduce_scalar.push_back(1);

    // Step 2: Subtract max + Exp (fused)
    dfb_reduce_scalar.wait_front(1);

    tile_regs_acquire();
    sub_bcast_cols_init(dfb::softmax_tmp, dfb::reduce_scalar);
    sub_tiles_bcast_cols(dfb::softmax_tmp, dfb::reduce_scalar, 0, 0, 0);
    exp_tile_init</*APPROX=*/1>();
    exp_tile</*APPROX=*/1>(0);
    tile_regs_commit();

    dfb_softmax_tmp.pop_front(1);
    dfb_softmax_tmp.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, dfb::softmax_tmp);
    tile_regs_release();
    dfb_softmax_tmp.push_back(1);

    dfb_reduce_scalar.pop_front(1);

    // Step 3: Reduce SUM per row + reciprocal
    dfb_softmax_tmp.wait_front(1);
    dfb_reduce_scalar.reserve_back(1);

    tile_regs_acquire();
    reconfig_data_format(dfb::bcast_scaler, dfb::softmax_tmp);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::softmax_tmp, dfb::bcast_scaler, dfb::reduce_scalar);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb::softmax_tmp, dfb::bcast_scaler, 0, 0, 0);
    reduce_uninit(dfb::reduce_scalar);
    recip_tile_init();
    recip_tile(0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, dfb::reduce_scalar);
    tile_regs_release();
    dfb_reduce_scalar.push_back(1);

    // Step 4: Multiply by 1/sum + copy indices (fused, one DST cycle)
    dfb_softmax_tmp.wait_front(1);
    dfb_reduce_scalar.wait_front(1);
    dfb_intermed_val.wait_front(1);
    dfb_final_out.reserve_back(2);

    tile_regs_acquire();
    mul_bcast_cols_init(dfb::softmax_tmp, dfb::reduce_scalar);
    mul_tiles_bcast<BroadcastType::COL>(dfb::softmax_tmp, dfb::reduce_scalar, 0, 0, 0);

    copy_init(dfb::intermed_val);
    copy_tile(dfb::intermed_val, 0, 1);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, dfb::final_out);  // softmax weights
    pack_tile(1, dfb::final_out);  // indices
    tile_regs_release();

    dfb_final_out.push_back(2);
    dfb_softmax_tmp.pop_front(1);
    dfb_reduce_scalar.pop_front(1);
    dfb_intermed_val.pop_front(1);

    dfb_softmax_mask.pop_front(1);
    dfb_bcast_scaler.pop_front(1);
}
