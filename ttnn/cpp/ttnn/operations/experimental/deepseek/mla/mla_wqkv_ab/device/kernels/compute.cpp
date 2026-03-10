// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/api/tt-metalium/constants.hpp"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"

#include "api/compute/tile_move_copy.h"
#include "rope_sfpu.h"

void kernel_main() {
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_addr = get_arg_val<uint32_t>(argidx++);
    const auto rope_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto pos = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_r2c_rope = tt::CBIndex::c_3;
    constexpr auto cb_s2c_out = tt::CBIndex::c_4;

    // Constants for MLA WqkvAb
    constexpr uint32_t k_tiles = 7168 / 32;
    constexpr uint32_t n_tiles_this_core = 6;

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = 6;
    constexpr uint32_t w_tiles_per_txn = 7;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    constexpr uint32_t num_w_tiles = k_tiles * n_tiles_this_core;
    const uint32_t w_num_blocks = num_w_tiles / w_tiles_per_block;

    //-------------------------------------------------------------------------
    // Compute configuration
    //-------------------------------------------------------------------------
    // Pack is configured to Float16_b
    pack_reconfig_data_format(cb_s2c_out);

    // Unpacker B is for input/activation, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in);

    // Unpacker A is for weight, so Bfp8_b
    reconfig_data_format_srca(cb_r2c_w);

    // Initialize matmul: input @ weight -> output
    mm_block_init(
        cb_s2c_in, cb_r2c_w, cb_s2c_out, /*transpose=*/false, /*ct_dim=*/n_tiles_this_core, /*rt_dim=*/1, /*kt_dim=*/1);

    //-------------------------------------------------------------------------
    // Compute: input @ weight -> output
    //-------------------------------------------------------------------------
    tile_regs_acquire();

    uint32_t in0_index = 0;

    for (uint32_t block_id = 0; block_id < w_num_blocks; ++block_id) {
        cb_wait_front(cb_r2c_w, w_tiles_per_block);

        // Process each block in ct_dim chunks, similar to matmul_wo.
        for (uint32_t k = 0; k < w_tiles_per_block; k += n_tiles_this_core) {
            matmul_block(
                cb_s2c_in,
                cb_r2c_w,
                /*in0_index=*/in0_index++,
                /*in1_tile_index=*/k,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/n_tiles_this_core,
                /*rt_dim=*/1,
                /*kt_dim=*/1);
        }

        cb_pop_front(cb_r2c_w, w_tiles_per_block);
    }

    cb_wait_front(cb_r2c_rope, 1);

    // Configure unpacker A to Float16_b
    reconfig_data_format_srca(cb_r2c_rope);

    if (dram_bank_id == 11) {
        copy_tile_init(cb_r2c_rope);
        copy_tile(cb_r2c_rope, 0, n_tiles_this_core);
        rope_tile_init();

        // This is the first tile with k_pe output
        rope_tile(n_tiles_this_core - 2);
    }
    cb_pop_front(cb_r2c_rope, 1);

    tile_regs_commit();
    tile_regs_wait();

    pack_tile_block(0, cb_s2c_out, n_tiles_this_core);
    tile_regs_release();

    // We have one extra slot reserved, which we won't use, drain it.
    cb_wait_front(cb_r2c_w, w_tiles_per_block);
    cb_pop_front(cb_r2c_w, w_tiles_per_block);
}
