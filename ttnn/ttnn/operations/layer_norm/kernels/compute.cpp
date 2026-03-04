// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Compute Kernel (Stage 2: subtract_mean)
//
// Per tile-row:
//   Phase 0: Tilize cb_in (RM) -> cb_tilize_out (tiled)
//   Phase 1: Reduce SUM (with scaler=1/W) -> cb_mean (row mean)
//   Phase 2: Sub broadcast COL (x - mean) -> cb_out
//   Phase 8: Untilize cb_out -> cb_untilize_out (RM sticks for writer)
//
// Compile-time args:
//   [0]  Ht         -- tile-rows to process (N / 32)
//   [1]  Wt         -- tiles per row (W / 32)
//   [2]  has_weight -- 1 if gamma provided
//   [3]  has_bias   -- 1 if beta provided

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

// CB indices matching op_design.md
constexpr uint32_t cb_in = 0;             // RM input pages (reader -> compute)
constexpr uint32_t cb_eps = 1;            // epsilon tile
constexpr uint32_t cb_scaler = 2;         // reduce scaler (1/W)
constexpr uint32_t cb_weight = 3;         // gamma tiles
constexpr uint32_t cb_bias = 4;           // beta tiles
constexpr uint32_t cb_tilize_out = 16;    // tilize output
constexpr uint32_t cb_out = 17;           // final output (pre-untilize)
constexpr uint32_t cb_untilize_out = 18;  // untilized output (RM, for writer)
constexpr uint32_t cb_mean = 24;          // row mean
constexpr uint32_t cb_x_minus_mean = 25;  // x - mean
constexpr uint32_t cb_sq = 26;            // (x - mean)^2
constexpr uint32_t cb_var = 27;           // variance
constexpr uint32_t cb_inv_std = 28;       // 1/sqrt(var + eps)
constexpr uint32_t cb_norm = 29;          // normalized output

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_weight = get_compile_time_arg_val(2);
    constexpr uint32_t has_bias = get_compile_time_arg_val(3);

    // Hardware startup: initialize unpack/math/pack pipelines.
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    for (uint32_t tile_row = 0; tile_row < Ht; ++tile_row) {
        // Phase 0: Tilize — convert Wt RM pages from cb_in into Wt tiles in cb_tilize_out.
        compute_kernel_lib::tilize<cb_in, cb_tilize_out>(Wt, 1);

        // Phase 1: Compute mean — reduce SUM with scaler=1/W.
        // WaitUpfrontNoPop: tiles persist in cb_tilize_out for Phase 2.
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilize_out, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 2: x - mean (broadcast COL subtract).
        // cb_tilize_out: already waited from Phase 1 (NoPop), consumed and popped at end.
        // cb_mean: waited and popped at end.
        // Output: cb_out (for stages 1-2, route directly to cb_out for untilize).
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_tilize_out, cb_mean, cb_out, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 8: Untilize cb_out -> cb_untilize_out.
        compute_kernel_lib::untilize<
            Wt,
            cb_out,
            cb_untilize_out,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitUpfront>(1);
    }
}
