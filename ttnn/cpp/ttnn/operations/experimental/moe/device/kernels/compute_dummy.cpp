// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "moe_ring_common.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w0_w1 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_3;
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_4;
    constexpr auto cb_c2w_out = tt::CBIndex::c_5;
    constexpr auto cb_w2s_out = tt::CBIndex::c_6;

    // CB Aliases
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;

    // Constants for MoE
    constexpr uint32_t num_w0_w1_tiles_h = moe_ring::NUM_W0_W1_TILES_H;
    constexpr uint32_t num_w2_tiles_h = moe_ring::NUM_W2_TILES_H;

    const uint32_t num_w0_w1_tiles_w = moe_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];
    const uint32_t num_w2_tiles_w = moe_ring::W2_TILES_PER_CORE_A[ring_core_id];

    const uint32_t num_elt_tiles = num_w0_w1_tiles_w;
    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    //-------------------------------------------------------------------------
    // W0 and W1 reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_txns_per_block = moe_ring::W0_W1_TXNS_PER_BLOCK;
    constexpr uint32_t w0_w1_tiles_per_txn = moe_ring::W0_W1_TILES_PER_TXN;
    constexpr uint32_t w0_w1_tiles_per_block = w0_w1_tiles_per_txn * w0_w1_txns_per_block;  // 14 * 2 = 28
    constexpr uint32_t w0_w1_blocks_per_elt_tile =
        2 * (num_w0_w1_tiles_h / w0_w1_tiles_per_txn) / w0_w1_txns_per_block;  // 16
    const uint32_t w0_w1_blocks_per_expert = moe_ring::W0_W1_BLOCKS_PER_EXPERT_A[ring_core_id];
    // 2 * num_w0_w1_tiles_w * num_w0_w1_tiles_h / w0_w1_tiles_per_block;  // (5|6 * 224) / 28 = 80|96

    // W2 reading constants
    constexpr uint32_t w2_txns_per_block = moe_ring::W2_TXNS_PER_BLOCK;
    constexpr uint32_t w2_tiles_per_txn = moe_ring::W2_TILES_PER_TXN;
    constexpr uint32_t w2_tiles_per_block = w2_tiles_per_txn * w2_txns_per_block;               // 14 * 2 = 28
    constexpr uint32_t w2_txns_h = (num_w2_tiles_h + w2_tiles_per_txn - 1) / w2_tiles_per_txn;  // 5 (round up)
    constexpr uint32_t w2_blocks_per_two_mm2_tile = 2 * w2_txns_h / w2_txns_per_block;          // 2 * 5 / 2 = 5
    const uint32_t w2_blocks_per_expert = moe_ring::W2_BLOCKS_PER_EXPERT_A[ring_core_id];
    // (num_w2_tiles_w/2) * w2_blocks_per_two_mm2_tile;  // (18|20 / 2) * 5 = 45|50

    //-------------------------------------------------------------------------
    // Ring setup
    //-------------------------------------------------------------------------
    // The number of times to repeat the all2all
    constexpr uint32_t num_a2a_iters = moe_ring::NUM_A2A_ITERS_A;

    // The number of steps to take in the all2all is the number of cores
    constexpr uint32_t num_a2a_steps_per_iter = moe_ring::NUM_CORES;

    // The number of tiles to send in each step
    // We send 6 tiles in each step, even though some cores in some steps may have only 5 valid ones
    constexpr uint32_t tiles_per_step = moe_ring::IN2_TILES_PER_STEP_A;  // max(num_w0_w1_tiles_w)

    //-------------------------------------------------------------------------
    // Dummy compute
    //-------------------------------------------------------------------------

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------

    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        // Read W0 and W1 from cb_r2c_w0_w1 and write final output to cb_c2w_elt
        for (uint32_t i = 0; i < num_elt_tiles; ++i) {
            for (uint32_t block_id = 0; block_id < w0_w1_blocks_per_elt_tile; ++block_id) {
                cb_wait_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);
                cb_pop_front(cb_r2c_w0_w1, w0_w1_tiles_per_block);
            }
        }

        // Signal to DM1 that the output from this core is ready
        cb_reserve_back(cb_c2w_rdy, 1);
        cb_push_back(cb_c2w_rdy, 1);

        // Read W2 from DRAM into CB
        for (uint32_t i = 0; i < (num_mm2_tiles >> 1); ++i) {
            uint32_t dm1_step = 0;
            uint32_t dm1_tiles_remaining = moe_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];
            cb_wait_front(cb_w2c_rdy, 1);

            for (uint32_t block_id = 0; block_id < w2_blocks_per_two_mm2_tile; ++block_id) {
                cb_wait_front(cb_r2c_w2, w2_tiles_per_block);

                for (uint32_t k = 0; k < w2_tiles_per_block; k += 2) {
                    // For cores which have only 18 mm2 tiles, we need to drain the pipeline for the last 2 and just
                    // exit.
                    if ((block_id == (w2_blocks_per_two_mm2_tile - 1)) && (k == 16)) {
                        cb_pop_front(cb_w2c_rdy, 1);
                        break;
                    }
                    if (dm1_tiles_remaining == 0) {
                        cb_pop_front(cb_w2c_rdy, 1);
                        cb_wait_front(cb_w2c_rdy, 1);
                        dm1_tiles_remaining = moe_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][++dm1_step];
                    }
                    dm1_tiles_remaining--;
                }
                cb_pop_front(cb_r2c_w2, w2_tiles_per_block);
            }

            // Signal to DM1 that we finished using this in2
            // Also serves to signal that we have packed 2 output tiles
            cb_reserve_back(cb_c2w_rdy, 1);
            cb_push_back(cb_c2w_rdy, 1);
        }

        // For cores which have only 18 mm2 tiles, we need to drain the pipeline for the last 2.
        if ((num_mm2_tiles >> 1) < num_a2a_iters) {
            for (uint32_t step = 0; step < num_a2a_steps_per_iter; ++step) {
                cb_wait_front(cb_w2c_rdy, 1);
                cb_pop_front(cb_w2c_rdy, 1);
            }
            cb_wait_front(cb_c2w_rdy, 1);
            cb_pop_front(cb_c2w_rdy, 1);
        }
    }  // end for (expert_id)

    // Drain the pipeline - the last dummy push
    cb_wait_front(cb_r2c_w2, w2_tiles_per_block);
    cb_pop_front(cb_r2c_w2, w2_tiles_per_block);
}
}  // namespace NAMESPACE
