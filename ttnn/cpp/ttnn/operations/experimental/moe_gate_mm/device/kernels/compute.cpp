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

    // Constants for MoE Gate MM
    constexpr uint32_t num_w_tiles_h = 224;
    constexpr uint32_t num_out_tiles_h = 1;

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

    // Wait for weight tiles
    cb_wait_front(cb_r2c_w, num_w_tiles_h);

    // Wait for input tile
    cb_wait_front(cb_s2c_in, 1);

    // Perform matmul: 1 input tile @ 224 weight tiles -> 1 output tile
    for (uint32_t k = 0; k < num_w_tiles_h; k += 1) {
        matmul_block(
            cb_s2c_in,
            cb_r2c_w,
            /*in0_index=*/0,
            /*in1_index=*/k,
            /*idst=*/0,
            /*transpose=*/false,
            /*ct_dim=*/1,
            /*rt_dim=*/1,
            /*kt_dim=*/1);
    }

    // Pop consumed tiles
    cb_pop_front(cb_s2c_in, 1);
    cb_pop_front(cb_r2c_w, num_w_tiles_h);

    tile_regs_commit();
    tile_regs_wait();

    // Pack output tile
    pack_tile(0, cb_s2c_out, /*output_tile_index=*/0);

    tile_regs_release();
}
}  // namespace NAMESPACE
