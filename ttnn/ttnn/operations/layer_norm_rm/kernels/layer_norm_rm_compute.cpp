// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Phase 1: Tilize c_0 -> c_24
// Phase 10: Untilize c_24 -> c_16 (passthrough for Stage 1)
// Intermediate phases (2-9) will be added in later TDD stages.

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

constexpr uint32_t c_0 = 0;    // RM input sticks (from reader)
constexpr uint32_t c_8 = 8;    // Reduce scaler (persistent)
constexpr uint32_t c_16 = 16;  // Untilized output (to writer)
constexpr uint32_t c_24 = 24;  // Tilized input

void kernel_main() {
    // Compile-time args
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // Hardware initialization - required before using helpers
    compute_kernel_hw_startup(c_0, c_8, c_16);

    // Per-row loop
    for (uint32_t row = 0; row < num_rows_per_core; row++) {
        // Phase 1: Tilize c_0 -> c_24
        compute_kernel_lib::tilize<
            c_0,
            c_24,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // Stage 1: passthrough - directly untilize c_24 -> c_16
        // Phase 10: Untilize
        compute_kernel_lib::untilize<
            Wt,
            c_24,
            c_16,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
    }
}
