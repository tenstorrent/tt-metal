// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax - Compute Kernel
// Stage 2 (exp_only): Copy tile from c_0, apply exp_tile, pack to c_16

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/exp.h"

constexpr uint32_t cb_input = tt::CBIndex::c_0;
constexpr uint32_t cb_scaler = tt::CBIndex::c_1;
constexpr uint32_t cb_output = tt::CBIndex::c_16;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t num_rows_or_cols = get_compile_time_arg_val(0);
    constexpr uint32_t inner_dim = get_compile_time_arg_val(1);

    const uint32_t num_tiles = num_rows_or_cols * inner_dim;

    // Initialize compute hardware
    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);
    copy_tile_init(cb_input);
    exp_tile_init<false>();  // exact exp (not approximate)

    for (uint32_t t = 0; t < num_tiles; ++t) {
        // Wait for input tile
        cb_wait_front(cb_input, 1);

        // Reserve output space
        cb_reserve_back(cb_output, 1);

        // Copy tile to DST, apply exp, pack to output
        tile_regs_acquire();
        copy_tile(cb_input, 0, 0);  // CB tile index 0 -> DST register 0
        exp_tile<false>(0);         // exact exp in-place on DST[0]
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_output);  // DST register 0 -> output CB
        tile_regs_release();

        // Pop input, push output
        cb_pop_front(cb_input, 1);
        cb_push_back(cb_output, 1);
    }
}
