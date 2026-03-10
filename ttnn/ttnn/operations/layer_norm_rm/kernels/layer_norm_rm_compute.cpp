// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Stage 1 (data_pipeline): tilize c_0 -> c_16, untilize c_16 -> c_17

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t num_blocks_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // CB indices
    constexpr uint32_t cb_input_rm = 0;    // c_0: RM sticks from reader
    constexpr uint32_t cb_scaler = 8;      // c_8: reduce scaler
    constexpr uint32_t cb_tilized = 16;    // c_16: tilized data
    constexpr uint32_t cb_output_rm = 17;  // c_17: untilized output for writer

    // Hardware init - must come first
    compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_tilized);

    for (uint32_t block = 0; block < num_blocks_per_core; block++) {
        // Phase 1: Tilize (c_0 -> c_16)
        compute_kernel_lib::tilize<cb_input_rm, cb_tilized>(Wt, 1);

        // Phase 10: Untilize (c_16 -> c_17) -- skip all normalization for stage 1
        compute_kernel_lib::untilize<Wt, cb_tilized, cb_output_rm>(1);
    }
}
