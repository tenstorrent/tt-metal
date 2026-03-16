// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 1: tilize + untilize (identity passthrough)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // ========== CB indices ==========
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scaler = 8;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_tilized = 24;

    // ========== Hardware startup ==========
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    // ========== Main loop ==========
    for (uint32_t ht = 0; ht < Ht; ht++) {
        // Phase 1: Tilize - cb_in (RM) -> cb_tilized (tiled)
        compute_kernel_lib::tilize<Wt, cb_in, cb_tilized>(1);

        // Stage 1: direct passthrough - untilize cb_tilized -> cb_out
        compute_kernel_lib::untilize<Wt, cb_tilized, cb_out>(1);
    }
}
