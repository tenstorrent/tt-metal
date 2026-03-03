// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
//
// Stage 1: tilize c_0 -> c_1, untilize c_1 -> c_16 (identity passthrough)
//
// Compile-time args:
//   [0] Wt: tiles per row (W / 32)
//   [1] has_gamma
//   [2] has_beta
//
// Runtime args:
//   [0] num_rows_per_core: tile-rows this core processes

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// CB indices
constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_tilized = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_out = 16;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);

    // Runtime args
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(0);

    if (num_rows_per_core == 0) {
        return;
    }

    // Hardware init: srcA = cb_in_rm (for tilize), srcB = cb_scaler (for future reduce), out = cb_out
    compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out);

    // Per tile-row loop
    for (uint32_t row = 0; row < num_rows_per_core; row++) {
        // Phase 1: Tilize cb_in_rm (c_0) -> cb_tilized (c_1)
        // Wt tiles per block, 1 block per tile-row
        compute_kernel_lib::tilize<cb_in_rm, cb_tilized>(Wt, 1);

        // Stage 1: identity passthrough -- untilize directly from cb_tilized -> cb_out
        compute_kernel_lib::untilize<Wt, cb_tilized, cb_out>(1);
    }
}
