// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Stage 1: tilize c_0 -> c_1, then untilize c_1 -> c_16 (passthrough)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// CB indices
constexpr uint32_t cb_input_rm = 0;    // c_0: RM sticks for tilize
constexpr uint32_t cb_tilized = 1;     // c_1: Tilized input tiles
constexpr uint32_t cb_scaler = 8;      // c_8: Reduce scaler (1/W)
constexpr uint32_t cb_eps = 9;         // c_9: Epsilon tile
constexpr uint32_t cb_output_rm = 16;  // c_16: Untilized RM output

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);

void kernel_main() {
    // Hardware startup: srcA=c_0, srcB=c_8, ocb=c_16
    compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_output_rm);

    // Runtime args
    const uint32_t nblocks = get_arg_val<uint32_t>(0);

    // Wait for epsilon tile (pushed by reader, but not used in stage 1)
    cb_wait_front(cb_eps, 1);

    // Main loop: process nblocks tile-rows
    for (uint32_t block = 0; block < nblocks; ++block) {
        // Phase 1: Tilize (RM -> tile) c_0 -> c_1
        compute_kernel_lib::
            tilize<cb_input_rm, cb_tilized, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);

        // Phase 10: Untilize (tile -> RM) c_1 -> c_16
        // Stage 1: passthrough, so untilize directly from c_1
        compute_kernel_lib::
            untilize<Wt, cb_tilized, cb_output_rm, compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit>(
                1);
    }

    // Pop epsilon tile
    cb_pop_front(cb_eps, 1);
}
