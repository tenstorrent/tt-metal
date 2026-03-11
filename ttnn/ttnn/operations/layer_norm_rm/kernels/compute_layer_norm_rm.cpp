// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 1 (data_pipeline): tilize c_0 -> c_1, untilize c_1 -> c_17

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

constexpr uint32_t c_0 = 0;    // RM input staging for tilize
constexpr uint32_t c_1 = 1;    // Tilized input tiles
constexpr uint32_t c_2 = 2;    // Reduce scaler (unused in stage 1)
constexpr uint32_t c_17 = 17;  // Untilized RM output

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);

void kernel_main() {
    // Runtime args
    uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    // Hardware init - must come first
    compute_kernel_hw_startup(c_0, c_2, c_1);

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        // Phase 1: Tilize c_0 -> c_1
        compute_kernel_lib::tilize<
            c_0,
            c_1,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(Wt, 1);

        // Phase 13: Untilize c_1 -> c_17 (direct passthrough for stage 1)
        compute_kernel_lib::untilize<
            Wt,
            c_1,
            c_17,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
    }
}
