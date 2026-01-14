// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto core_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w0 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2c_mm0 = tt::CBIndex::c_2;
    constexpr auto cb_c2c_mm1 = tt::CBIndex::c_3;
    constexpr auto cb_c2w_elt = tt::CBIndex::c_4;
    constexpr auto cb_r2c_in2 = tt::CBIndex::c_5;
    constexpr auto cb_c2w_mm2 = tt::CBIndex::c_6;

    // CB Aliases
    constexpr auto cb_r2c_w1 = tt::CBIndex::c_0;
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;

    // Constants for MoE
    constexpr uint32_t num_w0_w1_tiles_h = 224;
    constexpr uint32_t num_w2_tiles_h = 64;

    const uint32_t num_w0_w1_tiles_w = (core_id < 8) ? 5 : 6;
    const uint32_t num_w2_tiles_w = (core_id < 8) ? 19 : 18;

    const uint32_t num_elt_tiles = num_w0_w1_tiles_w;
    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    // Pack is always configured to Float16_b
    pack_reconfig_data_format(cb_c2c_mm0);

    // Unpacker B is for input/activation and eltiwse inputs, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in);

    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        // Unpacker A is for W0,W1 and W2, so Bf4_b
        reconfig_data_format_srca(cb_r2c_w0);
        mm_init(cb_s2c_in, cb_r2c_w0, cb_c2c_mm0);

        //---------------------------------------------------------------------
        // Compute in @ W0
        //---------------------------------------------------------------------
        tile_regs_acquire();
        for (uint32_t k_idx = 0; k_idx < num_w0_w1_tiles_h; ++k_idx) {
            for (uint32_t dst_idx = 0; dst_idx < num_w0_w1_tiles_w; ++dst_idx) {
                cb_wait_front(cb_r2c_w0, 1);
                matmul_tiles(cb_s2c_in, cb_r2c_w0, k_idx, 0, dst_idx);
                cb_pop_front(cb_r2c_w0, 1);
            }
        }

        //---------------------------------------------------------------------
        // Apply SILU activation
        //---------------------------------------------------------------------
        silu_tile_init();
        for (uint32_t dst_idx = 0; dst_idx < num_w0_w1_tiles_w; ++dst_idx) {
            silu_tile(dst_idx);
        }
        tile_regs_commit();

        // cb_reserve_back(cb_c2c_mm0, num_w0_w1_tiles_w);
        tile_regs_wait();

        for (uint32_t dst_idx = 0; dst_idx < num_w0_w1_tiles_w; ++dst_idx) {
            pack_tile(dst_idx, cb_c2c_mm0);
        }

        tile_regs_release();
        // cb_push_back(cb_c2c_mm0, num_w0_w1_tiles_w);

        //---------------------------------------------------------------------
        // Compute in @ W1
        //---------------------------------------------------------------------
        tile_regs_acquire();
        for (uint32_t k_idx = 0; k_idx < num_w0_w1_tiles_h; ++k_idx) {
            for (uint32_t dst_idx = 0; dst_idx < num_w0_w1_tiles_w; ++dst_idx) {
                cb_wait_front(cb_r2c_w1, 1);
                matmul_tiles(cb_s2c_in, cb_r2c_w1, k_idx, 0, dst_idx);
                cb_pop_front(cb_r2c_w1, 1);
            }
        }

        //---------------------------------------------------------------------
        // Eltwise multiply each dst with its corresponding tile from mm0
        //---------------------------------------------------------------------
        reconfig_data_format_srca(cb_c2c_mm0);
        binary_op_init_common(cb_c2c_mm0, cb_c2c_mm0, cb_c2w_elt);
        mul_tiles_init(cb_c2c_mm0, cb_c2c_mm0);
        // cb_wait_front(cb_c2c_mm0, num_w0_w1_tiles_w);

        for (uint32_t dst_idx = 0; dst_idx < num_w0_w1_tiles_w; ++dst_idx) {
            // TODO: Reuse dst for operand B and write in place
            mul_tiles(cb_c2c_mm0, cb_c2c_mm0, dst_idx, dst_idx, dst_idx);
        }
        tile_regs_commit();
        // cb_pop_front(cb_c2c_mm0, num_w0_w1_tiles_w);

        tile_regs_wait();
        cb_reserve_back(cb_c2w_elt, num_w0_w1_tiles_w);
        for (uint32_t dst_idx = 0; dst_idx < num_w0_w1_tiles_w; ++dst_idx) {
            pack_tile(dst_idx, cb_c2w_elt);
        }
        cb_push_back(cb_c2w_elt, num_w0_w1_tiles_w);
        tile_regs_release();

        //---------------------------------------------------------------------
        // Compute in @ W2
        //---------------------------------------------------------------------
        // TODO: Implement W2 computation
        //---------------------------------------------------------------------
        reconfig_data_format_srca(cb_r2c_w2);
        mm_init(cb_s2c_in, cb_r2c_w2, cb_c2w_mm2);
        tile_regs_acquire();
        for (uint32_t i = 0; i < num_w2_tiles_h; ++i) {
            cb_wait_front(cb_r2c_in2, 1);
            for (uint32_t j = 0; j < num_w2_tiles_w; ++j) {
                cb_wait_front(cb_r2c_w2, 1);
                // TODO: Need to pack existing partial sum and put it back
                // to accumulate here.
                matmul_tiles(cb_s2c_in, cb_r2c_w2, 0, 0, 0);
                cb_pop_front(cb_r2c_w2, 1);
            }
            cb_pop_front(cb_r2c_in2, 1);
        }
        tile_regs_commit();

        // Pop out excess tiles
        const uint32_t excess_tiles = 14 - ((num_w2_tiles_h * num_w2_tiles_w) % 14);
        for (uint32_t i = 0; i < excess_tiles; ++i) {
            cb_wait_front(cb_r2c_w2, 1);
            cb_pop_front(cb_r2c_w2, 1);
        }

        tile_regs_wait();
        for (uint32_t idx = 0; idx < num_mm2_tiles; ++idx) {
            cb_reserve_back(cb_c2w_mm2, 1);
            pack_tile(0, cb_c2w_mm2);
            cb_push_back(cb_c2w_mm2, 1);
        }
        tile_regs_release();

    }  // end for (expert_id)
}
}  // namespace NAMESPACE
