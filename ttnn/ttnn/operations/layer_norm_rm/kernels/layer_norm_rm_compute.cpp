// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Stage 1 (data_pipeline): tilize input (c_0 -> c_1), then untilize (c_1 -> c_16)
// Later stages add: reduce_mean, subtract_mean, variance_inv_std, affine

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    // ========== COMPILE-TIME ARGS ==========
    constexpr uint32_t nblocks_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // ========== CB CONSTANTS ==========
    constexpr uint32_t cb_input_rm = 0;    // c_0: input RM sticks
    constexpr uint32_t cb_tilized = 1;     // c_1: tilized input tiles
    constexpr uint32_t cb_output_rm = 16;  // c_16: output RM sticks

    // ========== HW STARTUP ==========
    // 3-arg form: srcA=cb_input_rm, srcB=cb_input_rm (not used for tilize), ocb=cb_output_rm
    compute_kernel_hw_startup(cb_input_rm, cb_input_rm, cb_output_rm);

    // ========== MAIN LOOP ==========
    for (uint32_t block = 0; block < nblocks_per_core; block++) {
        // Phase 1: Tilize input (c_0 -> c_1)
        compute_kernel_lib::tilize<cb_input_rm, cb_tilized>(Wt, 1);

        // Stage 1 bypass: directly untilize tilized data to output
        // Phase 10: Untilize (c_1 -> c_16)
        compute_kernel_lib::untilize<Wt, cb_tilized, cb_output_rm>(1);
    }
}
