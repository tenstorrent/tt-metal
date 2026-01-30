// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto partial_semaphore = get_arg_val<uint32_t>(argidx++);
    const auto send_core = get_arg_val<uint32_t>(argidx++);
    const auto neighbor1_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor1_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto neighbor2_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto neighbor2_physical_y = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_in2 = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;

    // NOC Packet size
    constexpr uint32_t noc_packet_size = 8192;

    // Constants for MoE Gate MM
    const uint32_t num_w_tiles_h = send_core ? 2 * 72 : 2 * 76;
    constexpr uint32_t num_w_tiles_w = 1;

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = 8;
    constexpr uint32_t w_tiles_per_txn = noc_packet_size / 2048;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    const uint32_t w_num_blocks = num_w_tiles_h * num_w_tiles_w / w_tiles_per_block;
    const uint32_t w_last_block_txns = ((num_w_tiles_h * num_w_tiles_w) % w_tiles_per_block) / w_tiles_per_txn;
    const uint32_t w_tiles_per_block_last = w_last_block_txns * w_tiles_per_txn;

    //-------------------------------------------------------------------------
    // Compute configuration
    //-------------------------------------------------------------------------
    // Pack is configured to Float16_b
    pack_reconfig_data_format(cb_s2c_out);

    // Unpacker B is for input/activation, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in);

    // Unpacker A is for weight, so Float16_b
    reconfig_data_format_srca(cb_r2c_w);

    if (send_core) {
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
    } else {
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
        cb_pop_front(cb_r2c_w, w_tiles_per_block);

        binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w2c_in2);

        // Wait for the partial to come, add it
        cb_wait_front(cb_w2c_in2, 1);
        binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w2c_in2, 0, 0);
        cb_pop_front(cb_w2c_in2, 1);

        tile_regs_commit();
        tile_regs_wait();
        // Pack output tile
        pack_tile(0, cb_s2c_out);
        tile_regs_release();

        // Signal to DM1 that we have finished
        cb_reserve_back(cb_c2w_rdy, 1);
        cb_push_back(cb_c2w_rdy, 1);
    }
}
