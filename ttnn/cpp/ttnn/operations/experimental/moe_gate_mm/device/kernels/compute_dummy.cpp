// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"

void kernel_main() {
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t reduce_core_id = get_named_compile_time_arg_val("reduce_core_id");
    constexpr uint32_t reduce_core_physical_x = get_named_compile_time_arg_val("reduce_core_physical_x");
    constexpr uint32_t reduce_core_physical_y = get_named_compile_time_arg_val("reduce_core_physical_y");

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
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;

    // NOC Packet size
    constexpr uint32_t noc_packet_size = 8192;

    // Constants for MoE Gate MM
    constexpr uint32_t num_w_tiles_h = 20;
    constexpr uint32_t num_w_tiles_w = 8;

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = 8;
    constexpr uint32_t w_tiles_per_txn = noc_packet_size / 2048;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    constexpr uint32_t w_num_blocks = num_w_tiles_h / w_tiles_per_block;

    //-------------------------------------------------------------------------
    // Dummy compute
    //-------------------------------------------------------------------------
    for (uint32_t block_id = 0; block_id < w_num_blocks; ++block_id) {
        cb_wait_front(cb_r2c_w, w_tiles_per_block);
        cb_pop_front(cb_r2c_w, w_tiles_per_block);
    }

    // Signal to DM1 that we have finished
    cb_reserve_back(cb_c2w_rdy, 1);
    cb_push_back(cb_c2w_rdy, 1);

    // Drain the pipeline - the last dummy push
    cb_wait_front(cb_r2c_w, w_tiles_per_block);
    cb_pop_front(cb_r2c_w, w_tiles_per_block);
}
