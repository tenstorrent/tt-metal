// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_s2c_out = tt::CBIndex::c_2;

    // NOC Packet size
    constexpr uint32_t noc_packet_size = 8192;

    // Constants for MoE Gate MM
    constexpr uint32_t num_w_tiles_h = 224;
    constexpr uint32_t num_out_tiles_h = 1;

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = 8;
    constexpr uint32_t w_tiles_per_txn = noc_packet_size / 2048;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    constexpr uint32_t w_num_blocks = num_w_tiles_h / w_tiles_per_block;

    //-------------------------------------------------------------------------
    // Compute configuration
    //-------------------------------------------------------------------------
    // Pack is configured to Float16_b
    pack_reconfig_data_format(cb_s2c_out);

    // Unpacker B is for input/activation, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in);

    // Unpacker A is for weight, so Float16_b
    reconfig_data_format_srca(cb_r2c_w);

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
    tile_regs_commit();

    tile_regs_wait();

    // Pack output tile
    pack_tile(0, cb_s2c_out);

    tile_regs_release();

    // Drain the pipeline - the last dummy push
    cb_wait_front(cb_r2c_w, w_tiles_per_block);
    cb_pop_front(cb_r2c_w, w_tiles_per_block);
}
}  // namespace NAMESPACE
