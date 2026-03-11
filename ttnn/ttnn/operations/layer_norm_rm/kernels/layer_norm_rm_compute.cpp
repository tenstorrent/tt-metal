// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Stage 2: tilize, reduce_row(mean via 1/W scaler), sub(COL), square, untilize

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
// CB indices
constexpr uint32_t cb_rm_input = 0;    // RM input sticks
constexpr uint32_t cb_scaler = 8;      // Reduce scaler (1/W)
constexpr uint32_t cb_rm_output = 16;  // RM output sticks
constexpr uint32_t cb_tilized = 24;    // Tilized input / reused intermediate
constexpr uint32_t cb_mean = 25;       // Row mean (1 tile)
constexpr uint32_t cb_centered = 26;   // x - mean (Wt tiles)
constexpr uint32_t cb_squared = 27;    // (x - mean)^2 (Wt tiles)

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t max_blocks = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);

void kernel_main() {
    // Hardware startup
    compute_kernel_hw_startup(cb_tilized, cb_scaler, cb_rm_output);

    // Get actual num_blocks for this core from runtime args
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // Wait for scaler CB (persists entire program)
    // Scaler contains 1/W, so reduce<SUM> with this scaler computes mean directly
    cb_wait_front(cb_scaler, 1);

    // Main loop: per block
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Phase 1: Tilize (c_0 -> c_24)
        compute_kernel_lib::
            tilize<cb_rm_input, cb_tilized, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);

        // Phase 2: Reduce Row for mean (c_24, c_8 -> c_25)
        // Scaler CB has 1/W; reduce<SUM> computes: sum(x_i * (1/W)) = mean
        // WaitUpfrontNoPop: c_24 tiles persist for Phase 3
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 3: Sub mean with COL broadcast (c_24 - c_25 -> c_26)
        // c_24: already waited from Phase 2, NoWaitNoPop -> caller pops
        // c_25: freshly pushed by Phase 2, WaitUpfrontPopAtEnd -> helper waits+pops
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        cb_pop_front(cb_tilized, Wt);  // Manual pop for NoWaitNoPop A

        // Phase 4: Square centered (c_26 -> c_27)
        // WaitUpfrontNoPop: c_26 tiles persist for Phase 7 (normalize)
        // For this stage, we untilize from c_27 (squared output)
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_squared, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Untilize from c_27 (squared output)
        compute_kernel_lib::
            untilize<Wt, cb_squared, cb_rm_output, compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit>(
                1);
    }

    // Pop persistent scaler
    cb_pop_front(cb_scaler, 1);
}
