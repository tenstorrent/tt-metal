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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

// CB indices
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_scaler = 8;
constexpr uint32_t cb_out_rm = 16;
constexpr uint32_t cb_input_tiled = 24;
constexpr uint32_t cb_mean = 25;
constexpr uint32_t cb_centered = 26;

void kernel_main() {
    constexpr uint32_t num_rows = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // Hardware init: srcA=cb_input_rm, srcB=cb_scaler, output=cb_out_rm
    compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_out_rm);

    // Wait for scaler tile (1/W) - loaded once, never popped
    cb_wait_front(cb_scaler, 1);

    // Per-row loop
    for (uint32_t row = 0; row < num_rows; ++row) {
        // Phase 0: Tilize input from cb_input_rm to cb_input_tiled
        compute_kernel_lib::tilize<cb_input_rm, cb_input_tiled>(Wt, 1);

        // Phase 1: Reduce mean across W dimension
        // SUM reduction with 1/W scaler gives mean
        // WaitUpfrontNoPop: c_24 tiles persist for Phase 2
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_input_tiled, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 2: Subtract mean from input (broadcast COL)
        // c_24 already waited from Phase 1 (NoWaitNoPop)
        // c_25 waited and popped at end (WaitUpfrontPopAtEnd)
        // Output to c_26 (Bulk push)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_input_tiled, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        // Manual pop of c_24 after sub (it was kept via NoPop)
        cb_pop_front(cb_input_tiled, Wt);

        // Stage 2: Untilize centered output to cb_out_rm
        compute_kernel_lib::untilize<Wt, cb_centered, cb_out_rm>(1);
    }
}
