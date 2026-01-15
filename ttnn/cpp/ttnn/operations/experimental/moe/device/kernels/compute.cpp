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
    const uint32_t num_w2_tiles_w = (core_id < 8) ? 18 : 20;

    const uint32_t num_elt_tiles = num_w0_w1_tiles_w;
    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    // W0 and W1 reading constants
    constexpr uint32_t w0_w1_tiles_per_txn = 14;
    const uint32_t w0_w1_txns = (core_id < 8) ? 2 * 80 : 2 * 96;
    // num_w0_w1_tiles_w * num_w0_w1_tiles_h / w0_w1_tiles_per_txn;  // (5|6 * 224) / 14 = 80|96

    // W2 reading constants
    // Total tiles of w2 does not divide evenly by w2_tiles_per_txn (14), so we do it in two steps
    constexpr uint32_t w2_tiles_per_txn = 14;
    constexpr uint32_t w2_txns_h = (num_w2_tiles_h + w2_tiles_per_txn - 1) / w2_tiles_per_txn;  // 5 (round up)
    const uint32_t w2_txns = w2_txns_h * num_w2_tiles_w;

    constexpr uint32_t w0_w1_txns_per_elt_tile = 2 * (num_w0_w1_tiles_h / w0_w1_tiles_per_txn);

    //-------------------------------------------------------------------------
    // Compute
    //-------------------------------------------------------------------------
    // Pack is always configured to Float16_b
    pack_reconfig_data_format(cb_c2c_mm0);

    // Unpacker B is for input/activation and eltiwse inputs, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in);

    // Unpacker A is for W0,W1 and W2, so Bf4_b
    reconfig_data_format_srca(cb_r2c_w0);

    // Initialize matmul for W0
    mm_block_init(cb_s2c_in, cb_r2c_w0, cb_c2c_mm0, /*transpose=*/false, /*ct_dim=*/2, /*rt_dim=*/1, /*kt_dim=*/1);

    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        //---------------------------------------------------------------------
        // Compute in @ W0
        //---------------------------------------------------------------------
        for (uint32_t i = 0; i < num_elt_tiles; ++i) {
            uint32_t in0_index = 0;
            tile_regs_acquire();
            for (uint32_t txn = 0; txn < w0_w1_txns_per_elt_tile; ++txn) {
                cb_wait_front(cb_r2c_w0, w0_w1_tiles_per_txn);

                for (uint32_t k = 0; k < w0_w1_tiles_per_txn; k += 2) {
                    matmul_block(
                        cb_s2c_in,
                        cb_r2c_w0,
                        in0_index++,
                        /*in1_index=*/k,
                        /*idst=*/0,
                        /*transpose=*/false,
                        /*ct_dim=*/2,
                        /*rt_dim=*/1,
                        /*kt_dim=*/1);
                }
                cb_pop_front(cb_r2c_w0, w0_w1_tiles_per_txn);
            }

            //---------------------------------------------------------------------
            // Apply SILU activation and then eltwise multiply
            //---------------------------------------------------------------------
            // TODO: Eltwise multiply output of SILU in dst0 with dst1 and store in dst2
            // silu_tile_init();
            // silu_tile(0);
            tile_regs_commit();

            tile_regs_wait();
            cb_reserve_back(cb_c2w_elt, 1);
            pack_tile(0, cb_c2w_elt);
            cb_push_back(cb_c2w_elt, 1);
            tile_regs_release();
        }

        //---------------------------------------------------------------------
        // Compute in @ W2
        //---------------------------------------------------------------------
        for (uint32_t i = 0; i < (num_mm2_tiles / 2); ++i) {
            // cb_wait_front(cb_r2c_in2, 1);
            // cb_pop_front(cb_r2c_in2, 1);
            uint32_t in0_index = 0;

            tile_regs_acquire();

            for (uint32_t j = 0; j < (2 * w2_txns_h); ++j) {
                cb_wait_front(cb_r2c_w2, w2_tiles_per_txn);
                for (uint32_t k = 0; k < w2_tiles_per_txn; k += 2) {
                    matmul_block(
                        cb_r2c_in2,
                        cb_r2c_w2,
                        in0_index++,
                        /*in1_index=*/k,
                        /*idst=*/0,
                        /*transpose=*/false,
                        /*ct_dim=*/2,
                        /*rt_dim=*/1,
                        /*kt_dim=*/1);
                }
                cb_pop_front(cb_r2c_w2, w2_tiles_per_txn);
            }

            tile_regs_commit();

            tile_regs_wait();
            cb_reserve_back(cb_c2w_mm2, 2);
            pack_tile_block(0, cb_c2w_mm2, 2);
            cb_push_back(cb_c2w_mm2, 2);
            tile_regs_release();
        }
    }  // end for (expert_id)
}
}  // namespace NAMESPACE
