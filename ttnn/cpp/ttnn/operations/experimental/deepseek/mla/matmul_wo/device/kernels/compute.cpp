// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo_ring_common.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"

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
    const uint32_t num_blocks_per_iter =
        (num_tiles_h * num_n_tiles_per_iter + w_tiles_per_block - 1) / w_tiles_per_block;
    const uint32_t w_total_blocks = num_blocks_per_iter * num_iters;

    //-------------------------------------------------------------------------
    // Compute
    //-------------------------------------------------------------------------
    // Pack is always configured to Float16_b
    pack_reconfig_data_format(cb_c2s_out);

    // Unpacker B is for input/activation and eltiwse inputs, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in);

    // Unpacker A is for W, so Bf8_b
    reconfig_data_format_srca(cb_r2c_w);

    // Initialize matmul
    mm_block_init(cb_s2c_in, cb_r2c_w, cb_c2s_out, /*transpose=*/false, /*ct_dim=*/7, /*rt_dim=*/1, /*kt_dim=*/1);

    //---------------------------------------------------------------------
    // Compute in @ W
    //---------------------------------------------------------------------
    uint32_t in0_index = 0;

    for (uint32_t iter_id = 0; iter_id < num_iters; ++iter_id) {
        tile_regs_acquire();
        for (uint32_t block_id = 0; block_id < num_blocks_per_iter; ++block_id) {
            cb_wait_front(cb_r2c_w, w_tiles_per_block);

            for (uint32_t k = 0; k < w_tiles_per_block; k += 7) {
                matmul_block(
                    cb_s2c_in,
                    cb_r2c_w,
                    in0_index++,
                    /*in1_tile_index=*/k,
                    /*idst=*/0,
                    /*transpose=*/false,
                    /*ct_dim=*/7,
                    /*rt_dim=*/1,
                    /*kt_dim=*/1);
            }
            cb_pop_front(cb_r2c_w, w_tiles_per_block);
        }

        tile_regs_commit();
        tile_regs_wait();

        for (uint32_t tile_id = 0; tile_id < num_n_tiles_per_iter; ++tile_id) {
            pack_tile(tile_id, cb_c2s_out);
        }
        tile_regs_release();

        // Signal to DM1 that 7 output tiles from this core are ready
        cb_reserve_back(cb_c2w_rdy, 1);
        cb_push_back(cb_c2w_rdy, 1);
    }

    // Drain the pipeline - the last dummy push
    cb_wait_front(cb_r2c_w, w_tiles_per_block);
    cb_pop_front(cb_r2c_w, w_tiles_per_block);
}
