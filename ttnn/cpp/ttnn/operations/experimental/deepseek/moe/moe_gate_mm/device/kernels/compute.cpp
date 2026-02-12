// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"

#include "bias_bcast_sfpu.h"
#include "top2_sum_sfpu.h"
#include "top4_sfpu.h"
#include "top8_sfpu.h"
#include "top8_merge_sfpu.h"

void kernel_main() {
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t collector_physical_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_physical_y = get_named_compile_time_arg_val("collector_physical_y");
    constexpr uint32_t first_physical_x = get_named_compile_time_arg_val("first_physical_x");
    constexpr uint32_t first_physical_y = get_named_compile_time_arg_val("first_physical_y");
    constexpr uint32_t column_id = get_named_compile_time_arg_val("column_id");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto partial_semaphore = get_arg_val<uint32_t>(argidx++);
    const auto is_send_core = get_arg_val<uint32_t>(argidx++);
    const auto neighbor1_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor1_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto neighbor2_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor2_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto core_id = get_arg_val<uint32_t>(argidx++);
    const auto raw_scores_semaphore = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_in2 = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;
    constexpr auto cb_w2c_in3 = tt::CBIndex::c_5;
    constexpr auto cb_w2c_in4 = tt::CBIndex::c_6;
    constexpr auto cb_w2c_in5 = tt::CBIndex::c_7;
    constexpr auto cb_w2c_in6 = tt::CBIndex::c_8;
    constexpr auto cb_w2c_in7 = tt::CBIndex::c_9;

    // Aliases
    constexpr auto cb_w2c_in8 = tt::CBIndex::c_6;

    // NOC Packet size
    constexpr uint32_t noc_packet_size = 8192;

    // Constants for MoE Gate MM
    const uint32_t num_w_tiles_h = is_send_core ? (2 * 72) : (2 * 76 + 1);
    constexpr uint32_t num_w_tiles_w = 1;

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = 8;
    constexpr uint32_t w_tiles_per_txn = noc_packet_size / 2048;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    const uint32_t w_num_blocks = num_w_tiles_h * num_w_tiles_w / w_tiles_per_block;
    const uint32_t w_remaining_tiles = (num_w_tiles_h * num_w_tiles_w) % w_tiles_per_block;
    const uint32_t w_last_block_txns = (w_remaining_tiles + w_tiles_per_txn - 1) / w_tiles_per_txn;  // Ceiling division
    const uint32_t w_tiles_per_block_last = w_remaining_tiles - 1;
    const uint32_t bias_tile_index = w_tiles_per_block_last;  // Bias is the last tile in the last block

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    constexpr uint32_t COLLECTOR_CORE_ID = 7;

    //-------------------------------------------------------------------------
    // Compute configuration
    //-------------------------------------------------------------------------
    // Pack is configured to Float16_b
    pack_reconfig_data_format(cb_s2c_out);

    // Unpacker B is for input/activation, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in);

    // Unpacker A is for weight, so Float16_b
    reconfig_data_format_srca(cb_r2c_w);

    if (is_send_core) {
        // Initialize matmul: input @ weight -> output
        mm_block_init(cb_s2c_in, cb_r2c_w, cb_s2c_out, /*transpose=*/false, /*ct_dim=*/2, /*rt_dim=*/1, /*kt_dim=*/1);

        //-------------------------------------------------------------------------
        // Compute: input @ 2 weights -> 2 outputss
        //-------------------------------------------------------------------------
        tile_regs_acquire();

        uint32_t tile_index = 2 * 76;
        for (uint32_t block_id = 0; block_id < w_num_blocks; ++block_id) {
            cb_wait_front(cb_r2c_w, w_tiles_per_block);

            for (uint32_t tile_id = 0; tile_id < w_tiles_per_block; tile_id += 2) {
                // Perform matmul: 1 input tile @ 2 weight tiles
                matmul_block(
                    cb_s2c_in,
                    cb_r2c_w,
                    /*in0_index=*/tile_index++,
                    /*in1_index=*/tile_id,
                    /*idst=*/0,
                    /*transpose=*/false,
                    /*ct_dim=*/2,
                    /*rt_dim=*/1,
                    /*kt_dim=*/1);
            }
            cb_pop_front(cb_r2c_w, w_tiles_per_block);
        }

        // Last block
        cb_wait_front(cb_r2c_w, w_tiles_per_block);
        for (uint32_t tile_id = 0; tile_id < w_tiles_per_block_last; tile_id += 2) {
            matmul_block(
                cb_s2c_in,
                cb_r2c_w,
                /*in0_index=*/tile_index++,
                /*in1_index=*/tile_id,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/2,
                /*rt_dim=*/1,
                /*kt_dim=*/1);
        }
        cb_pop_front(cb_r2c_w, w_tiles_per_block);

        tile_regs_commit();

        tile_regs_wait();

        cb_reserve_back(cb_c2w_rdy, 1);
        // Since neighbor1 is farther, we send it first (dst 1)
        pack_tile</*out_of_order_output=*/true>(1, cb_s2c_out, /*output_tile_index=*/0);
        cb_push_back(cb_c2w_rdy, 1);

        cb_reserve_back(cb_c2w_rdy, 1);
        pack_tile</*out_of_order_output=*/true>(0, cb_s2c_out, /*output_tile_index=*/0);
        cb_push_back(cb_c2w_rdy, 1);

        tile_regs_release();

        return;
    }

    // -------------------------------------------------------------------------
    // Rest of the 8 cores do more
    // -------------------------------------------------------------------------

    // Initialize matmul: input @ weight -> output
    mm_block_init(cb_s2c_in, cb_r2c_w, cb_s2c_out, /*transpose=*/false, /*ct_dim=*/1, /*rt_dim=*/1, /*kt_dim=*/1);

    //-------------------------------------------------------------------------
    // Compute: input @ weight -> output
    //-------------------------------------------------------------------------
    tile_regs_acquire();

    uint32_t tile_index = 0;
    for (uint32_t block_id = 0; block_id < w_num_blocks; ++block_id) {
        cb_wait_front(cb_r2c_w, w_tiles_per_block);

        for (uint32_t tile_id = 0; tile_id < w_tiles_per_block; ++tile_id) {
            // Perform matmul: 1 input tile @ 1 weight tile
            matmul_block(
                cb_s2c_in,
                cb_r2c_w,
                /*in0_index=*/tile_index++,
                /*in1_index=*/tile_id,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/1,
                /*rt_dim=*/1,
                /*kt_dim=*/1);
        }
        cb_pop_front(cb_r2c_w, w_tiles_per_block);
    }

    // Last block
    cb_wait_front(cb_r2c_w, w_tiles_per_block);
    for (uint32_t tile_id = 0; tile_id < w_tiles_per_block_last; ++tile_id) {
        matmul_block(
            cb_s2c_in,
            cb_r2c_w,
            /*in0_index=*/tile_index++,
            /*in1_index=*/tile_id,
            /*idst=*/0,
            /*transpose=*/false,
            /*ct_dim=*/1,
            /*rt_dim=*/1,
            /*kt_dim=*/1);
    }

    binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w2c_in2);

    // Wait for the partial to come, add it
    cb_wait_front(cb_w2c_in2, 1);
    binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w2c_in2, 0, 0);
    cb_pop_front(cb_w2c_in2, 1);

    //-------------------------------------------------------------------------
    // Sigmoid
    //-------------------------------------------------------------------------

    // Sigmoid the output
    sigmoid_tile_init();
    sigmoid_tile(0);

    //-------------------------------------------------------------------------
    // Retain a copy
    //-------------------------------------------------------------------------
    // Retain this copy for final scores (raw scores)
    copy_dest_values_init();
    copy_dest_values(0, 1);

    //-------------------------------------------------------------------------
    // Add bias
    //-------------------------------------------------------------------------
    copy_tile_init(cb_r2c_w);
    copy_tile(cb_r2c_w, bias_tile_index, 2);
    add_bias_init();
    add_bias(0, 2);

    //-------------------------------------------------------------------------
    // Sum of top2 scores for this group
    //-------------------------------------------------------------------------
    // First, pack the output and bring it back as transposed
    tile_regs_commit();
    tile_regs_wait();

    // Store the bias adjusted scores for transpose
    cb_reserve_back(cb_s2c_out, 1);
    pack_tile</*out_of_order_output=*/true>(0, cb_s2c_out, /*output_tile_index=*/0);
    cb_push_back(cb_s2c_out, 1);

    // Store the raw scores for transpose
    cb_reserve_back(cb_w2c_in3, 1);
    pack_tile(1, cb_w2c_in3);
    cb_push_back(cb_w2c_in3, 1);

    tile_regs_release();

    tile_regs_acquire();

    // Transpose
    cb_wait_front(cb_s2c_out, 1);
    transpose_wh_init_short(cb_s2c_out);
    transpose_wh_tile(cb_s2c_out, 0, 0);

    // Sum the top-2 of the output
    sum_top2_tile_init();
    sum_top2_tile(0);

    tile_regs_commit();

    cb_reserve_back(cb_c2w_rdy, 1);

    tile_regs_wait();
    // Pack output tile
    pack_tile(0, cb_c2w_rdy);
    tile_regs_release();
    cb_push_back(cb_c2w_rdy, 1);

    cb_pop_front(cb_r2c_w, w_tiles_per_block);

    //-------------------------------------------------------------------------
    // Non-collector cores
    //-------------------------------------------------------------------------
    if (core_id != COLLECTOR_CORE_ID) {
        // Wait for the group masks
        cb_wait_front(cb_w2c_in5, 1);

        tile_regs_acquire();

        // Get the adjusted scores
        transpose_wh_init_short(cb_s2c_out);
        transpose_wh_tile(cb_s2c_out, 0, 0);
        cb_pop_front(cb_s2c_out, 1);

        // Get the group masks
        copy_tile_init(cb_w2c_in5);
        copy_tile(cb_w2c_in5, 0, 2);

        // Get top 8 from adjusted scores, and mask them
        top8_tile_init();
        top8_tile(/*tile_index=*/core_id, /*dst_index=*/0);

        cb_pop_front(cb_w2c_in5, 1);
        cb_reserve_back(cb_w2c_in8, 1);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_w2c_in8);
        tile_regs_release();
        cb_push_back(cb_w2c_in8, 1);
    }

    //-------------------------------------------------------------------------
    // Collector core
    //-------------------------------------------------------------------------
    if (core_id == COLLECTOR_CORE_ID) {
        // I am collecting, let us wait for everyone else to finish sending their data to me
        cb_wait_front(cb_w2c_in4, 1);

        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_w2c_in4);

        // Copy the group scores
        copy_tile(cb_w2c_in4, 0, 0);

        //-------------------------------------------------------------------------
        // Top 4 groups for each token
        //-------------------------------------------------------------------------
        top4_tile_init();
        top4_tile(0);

        // Pack this out for other cores to get the group masks
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_w2c_in5, 1);
        pack_tile</*out_of_order_output=*/true>(0, cb_w2c_in5, /*output_tile_index=*/0);
        tile_regs_release();
        cb_push_back(cb_w2c_in5, 1);

        // Get top 8 from adjusted scores, and mask them
        tile_regs_acquire();
        transpose_wh_init_short(cb_s2c_out);
        transpose_wh_tile(cb_s2c_out, 0, 0);
        cb_pop_front(cb_s2c_out, 1);

        copy_tile_init(cb_w2c_in5);
        copy_tile(cb_w2c_in5, 0, 2);

        top8_tile_init();
        top8_tile(/*tile_index=*/core_id, /*dst_index=*/0);

        cb_pop_front(cb_w2c_in4, 1);

        // Wait for sorted top-8 from all other cores
        cb_wait_front(cb_w2c_in6, 4);

        copy_tile_init(cb_w2c_in6);
        // Tile ID 0 has my own data, so we copy to 1-4
        copy_tile(cb_w2c_in6, 0, 1);
        copy_tile(cb_w2c_in6, 1, 2);
        copy_tile(cb_w2c_in6, 2, 3);
        copy_tile(cb_w2c_in6, 3, 4);

        top8_merge_init();
        top8_merge<column_id>();

        cb_pop_front(cb_w2c_in6, 4);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_s2c_out, 1);
        pack_tile</*out_of_order_output=*/true>(0, cb_s2c_out, /*output_tile_index=*/0);
        cb_push_back(cb_s2c_out, 1);
        tile_regs_release();

        // Let DM1 know that we are done
        cb_reserve_back(cb_c2w_rdy, 1);
        cb_push_back(cb_c2w_rdy, 1);
    }
}
