// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo_ring_common.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"

void kernel_main() {
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t collector_physical_x = get_named_compile_time_arg_val("collector_physical_x");
    constexpr uint32_t collector_physical_y = get_named_compile_time_arg_val("collector_physical_y");
    constexpr uint32_t reduce_semaphore_id = get_named_compile_time_arg_val("reduce_semaphore_id");

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

    // CB Aliases
    constexpr auto cb_c2s_out = tt::CBIndex::c_1;

    // Constants for the kernel
    constexpr uint32_t num_w_tiles_w = matmul_wo_ring::NUM_W_TILES_W;
    constexpr uint32_t num_n_tiles_per_iter = matmul_wo_ring::N_TILES_PER_ITER;
    constexpr uint32_t max_num_tiles_h = matmul_wo_ring::MAX_K_TILES_PER_CORE;
    const uint32_t num_tiles_h = matmul_wo_ring::K_TILES_PER_CORE_A[dram_bank_id];

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = matmul_wo_ring::W_TXNS_PER_BLOCK;
    constexpr uint32_t w_tiles_per_txn = matmul_wo_ring::W_TILES_PER_TXN;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    const uint32_t num_iters = num_w_tiles_w / num_n_tiles_per_iter;
    const uint32_t w_total_blocks = num_tiles_h * num_iters * num_n_tiles_per_iter / w_tiles_per_block;

    const uint32_t last_block_tiles = (num_tiles_h * num_n_tiles_per_iter) % w_tiles_per_block;
    const uint32_t last_block_txns = (last_block_tiles + w_tiles_per_txn - 1) / w_tiles_per_txn;

    //-------------------------------------------------------------------------
    // Dummy compute
    //-------------------------------------------------------------------------

    for (uint32_t block_id = 0; block_id < w_total_blocks; ++block_id) {
        cb_wait_front(cb_r2c_w, w_tiles_per_block);
        cb_pop_front(cb_r2c_w, w_tiles_per_block);
    }

    // Signal to DM1 that the output from this core is ready
    cb_reserve_back(cb_c2w_rdy, 1);
    cb_push_back(cb_c2w_rdy, 1);

    // Drain the pipeline - the last dummy push
    cb_wait_front(cb_r2c_w, w_tiles_per_block);
    cb_pop_front(cb_r2c_w, w_tiles_per_block);
}
