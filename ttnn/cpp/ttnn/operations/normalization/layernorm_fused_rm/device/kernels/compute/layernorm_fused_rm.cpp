// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/pack_untilize.h"  // Use pack_untilize for RM output
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"

// Compute kernel for layernorm_fused_rm
// SIMPLIFIED TEST: Just tilize and untilize to verify basic data flow

namespace NAMESPACE {
void MAIN {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);

    // Runtime args
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    // CB indices
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;
    constexpr uint32_t cb_in_tiled = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_4;
    constexpr uint32_t cb_beta_rm = tt::CBIndex::c_5;
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;

    constexpr uint32_t TILE_HEIGHT = 32;

    // Pop gamma and beta to maintain CB balance
    cb_wait_front(cb_gamma_rm, 1);
    cb_pop_front(cb_gamma_rm, 1);
    cb_wait_front(cb_beta_rm, 1);
    cb_pop_front(cb_beta_rm, 1);

    // Pop scaler and epsilon to maintain CB balance
    cb_wait_front(cb_scaler, 1);
    cb_pop_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    cb_pop_front(cb_eps, 1);

    // SIMPLIFIED TEST: Just tilize and untilize
    for (uint32_t row = 0; row < num_tile_rows; row++) {
        // Phase 1: Tilize input
        cb_wait_front(cb_in_rm, TILE_HEIGHT);
        cb_reserve_back(cb_in_tiled, Wt);

        tilize_init(cb_in_rm, Wt, cb_in_tiled);
        tilize_block(cb_in_rm, Wt, cb_in_tiled);
        tilize_uninit(cb_in_rm, cb_in_tiled);

        cb_push_back(cb_in_tiled, Wt);
        cb_pop_front(cb_in_rm, TILE_HEIGHT);

        // Phase 2: Pack untilize output (tiled -> row major)
        // pack_untilize reads tiled tiles and outputs row-major data
        cb_wait_front(cb_in_tiled, Wt);
        cb_reserve_back(cb_out_rm, TILE_HEIGHT);

        // Use pack_untilize_init and pack_untilize_block
        // block_ct_dim and full_ct_dim must be compile-time constants
        // For Wt=1, we use 1 explicitly
        pack_untilize_init<1, 1>(cb_in_tiled, cb_out_rm);
        pack_untilize_block<1, 1>(cb_in_tiled, 1, cb_out_rm);  // 1 = block_rt_dim (1 tile row)
        pack_untilize_uninit(cb_out_rm);

        cb_push_back(cb_out_rm, TILE_HEIGHT);
        cb_pop_front(cb_in_tiled, Wt);
    }
}
}  // namespace NAMESPACE
