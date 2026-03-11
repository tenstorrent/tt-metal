// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Stage 1: tilize (c_0 -> c_24), untilize (c_24 -> c_16)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// CB indices
constexpr uint32_t cb_rm_input = 0;    // RM input sticks
constexpr uint32_t cb_scaler = 8;      // Reduce scaler (future stages)
constexpr uint32_t cb_rm_output = 16;  // RM output sticks
constexpr uint32_t cb_tilized = 24;    // Tilized intermediate

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t max_blocks = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);

void kernel_main() {
    // Hardware startup: input CB0 = cb_tilized (srcA), input CB1 = cb_scaler (srcB), output CB = cb_rm_output
    compute_kernel_hw_startup(cb_tilized, cb_scaler, cb_rm_output);

    // Get actual num_blocks for this core from runtime args
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // Main loop: per block, tilize then untilize
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Phase 1: Tilize (c_0 -> c_24)
        compute_kernel_lib::
            tilize<cb_rm_input, cb_tilized, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);

        // Phase 10: Untilize (c_24 -> c_16)
        compute_kernel_lib::
            untilize<Wt, cb_tilized, cb_rm_output, compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit>(
                1);
    }
}
