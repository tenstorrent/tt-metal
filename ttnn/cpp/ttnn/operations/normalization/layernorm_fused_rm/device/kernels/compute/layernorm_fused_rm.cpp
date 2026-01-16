// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

// Stub compute kernel for layernorm_fused_rm
// This is a PASSTHROUGH stub that copies input to output
// Actual computation (tilize, layernorm, untilize) will be implemented in Stage 7

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
    constexpr uint32_t cb_gamma_tiled = tt::CBIndex::c_6;
    constexpr uint32_t cb_beta_tiled = tt::CBIndex::c_7;
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;

    copy_tile_init(cb_in_rm);

    // STUB: Consume gamma/beta RM, produce tiled gamma/beta (persistent)
    // Gamma/beta are 1D (one stick each)
    // For stub, just pop gamma_rm/beta_rm and push Wt tiles to gamma_tiled/beta_tiled
    cb_wait_front(cb_gamma_rm, 1);
    cb_reserve_back(cb_gamma_tiled, Wt);
    cb_push_back(cb_gamma_tiled, Wt);
    cb_pop_front(cb_gamma_rm, 1);

    cb_wait_front(cb_beta_rm, 1);
    cb_reserve_back(cb_beta_tiled, Wt);
    cb_push_back(cb_beta_tiled, Wt);
    cb_pop_front(cb_beta_rm, 1);

    // STUB: Consume scaler and epsilon (read once, used for all rows)
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    // STUB: Process each row
    for (uint32_t row = 0; row < num_tile_rows; row++) {
        // Wait for input RM sticks (32 sticks per tile row)
        cb_wait_front(cb_in_rm, 32);

        // Reserve space for tiled input
        cb_reserve_back(cb_in_tiled, Wt);

        // STUB: Tilize - just push Wt tiles (garbage data is fine for stub)
        cb_push_back(cb_in_tiled, Wt);
        cb_pop_front(cb_in_rm, 32);

        // STUB: LayerNorm computation
        // In Stage 7, this will be:
        // 1. Compute mean
        // 2. Center (x - mean)
        // 3. Compute variance
        // 4. Compute inv_std
        // 5. Normalize
        // 6. Apply gamma/beta
        //
        // For now, just wait for tiled input and produce output

        cb_wait_front(cb_in_tiled, Wt);

        // STUB: Untilize - produce RM output (32 sticks)
        cb_reserve_back(cb_out_rm, 32);

        // STUB: Push 32 sticks of garbage output
        cb_push_back(cb_out_rm, 32);
        cb_pop_front(cb_in_tiled, Wt);
    }

    // Pop scaler and epsilon after all rows processed
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_eps, 1);

    // Note: gamma_tiled and beta_tiled are persistent - never popped
}
}  // namespace NAMESPACE
