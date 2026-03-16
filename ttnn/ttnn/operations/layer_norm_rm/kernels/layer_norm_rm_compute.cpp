// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 2: tilize + reduce(mean) + sub(COL) + untilize

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // ========== CB indices ==========
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scaler = 8;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_tilized = 24;
    constexpr uint32_t cb_mean = 25;
    constexpr uint32_t cb_centered = 26;

    // ========== Hardware startup ==========
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    // ========== Main loop ==========
    for (uint32_t ht = 0; ht < Ht; ht++) {
        // Phase 1: Tilize - cb_in (RM) -> cb_tilized (tiled)
        compute_kernel_lib::tilize<Wt, cb_in, cb_tilized>(1);

        // Phase 2: Reduce row (mean) - cb_tilized -> cb_mean
        // WaitUpfrontNoPop: waits for Wt tiles in cb_tilized, does NOT pop (persist for phase 3)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 3: Subtract mean (centering) - cb_tilized - cb_mean -> cb_centered
        // A: cb_tilized (already waited from phase 2, pop at end)
        // B: cb_mean (1 tile, wait and pop per tile for COL broadcast)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Stage 2: untilize cb_centered -> cb_out
        compute_kernel_lib::untilize<Wt, cb_centered, cb_out>(1);
    }
}
