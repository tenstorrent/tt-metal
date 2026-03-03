// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Runs on math RISC-V core (TRISC), performs tile operations.
//
// Compile-time args:
//   [0] num_rows   - total tile-rows to process (N_outer * Ht)
//   [1] Wt         - tiles per row (W / 32)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// CB indices
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_scaler = 8;
constexpr uint32_t cb_out_rm = 16;
constexpr uint32_t cb_input_tiled = 24;

void kernel_main() {
    constexpr uint32_t num_rows = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // Hardware init: srcA=cb_input_rm, srcB=cb_scaler, output=cb_out_rm
    compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_out_rm);

    // Per-row loop: tilize input, then untilize (identity pass-through)
    for (uint32_t row = 0; row < num_rows; ++row) {
        // Phase 0: Tilize input from cb_input_rm to cb_input_tiled
        compute_kernel_lib::tilize<cb_input_rm, cb_input_tiled>(Wt, 1);

        // Stage 1: Identity pass-through - untilize directly from cb_input_tiled to cb_out_rm
        compute_kernel_lib::untilize<Wt, cb_input_tiled, cb_out_rm>(1);
    }
}
