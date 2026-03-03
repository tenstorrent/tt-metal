// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
//
// Stage 2: tilize, reduce SUM for mean, broadcast mean to full row, untilize
//
// Strategy for mean broadcast:
//   x - (x - mean) = mean
//   Step 1: tilize input
//   Step 2: reduce SUM with 1/W scaler -> mean (column vector)
//   Step 3: sub<COL>(tilized, mean) -> centered
//   Step 4: sub<NONE>(tilized, centered) -> mean broadcast to all tiles
//   Step 5: untilize
//
// Compile-time args:
//   [0] Wt: tiles per row (W / 32)
//   [1] has_gamma
//   [2] has_beta
//
// Runtime args:
//   [0] num_rows_per_core: tile-rows this core processes

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

// CB indices
constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_tilized = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_mean = 24;
constexpr uint32_t cb_centered = 25;
constexpr uint32_t cb_normed = 27;  // Scratch for mean broadcast output

void kernel_main() {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);

    // Runtime args
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(0);

    if (num_rows_per_core == 0) {
        return;
    }

    // Hardware init
    compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out);

    // Per tile-row loop
    for (uint32_t row = 0; row < num_rows_per_core; row++) {
        // Phase 1: Tilize cb_in_rm (c_0) -> cb_tilized (c_1)
        compute_kernel_lib::tilize<cb_in_rm, cb_tilized>(Wt, 1);

        // Phase 2: Reduce SUM for mean
        // WaitUpfrontNoPop: tiles in c_1 persist for reuse in phases 3 and 4
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: centered = tilized - mean (COL broadcast)
        // c_1: NoWaitNoPop (already waited by reduce, keep persistent for phase 4)
        // c_24 (mean): WaitAndPopPerTile (1 tile, broadcast as COL, consumed after)
        // Output c_25: Bulk
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,        // c_1 already waited
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,  // c_24 mean: 1 tile
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 4: mean_broadcast = tilized - centered
        // c_1: NoWaitPopAtEnd (already waited, pop all Wt at end)
        // c_25 (centered): WaitUpfrontPopAtEnd (Wt tiles, consumed)
        // Output c_27: Bulk
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::NONE,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,       // c_1: pop all Wt at end
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,  // c_25: wait all, pop at end
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_tilized, cb_centered, cb_normed, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 5: Untilize cb_normed -> cb_out
        compute_kernel_lib::untilize<Wt, cb_normed, cb_out>(1);
    }
}
