// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Compute Kernel
// Runs on math RISC-V core, performs FPU/SFPU operations
//
// Stage 2 (square_reduce): tilize -> square -> reduce_row -> untilize
//   Outputs mean(x^2) per row with reduced shape

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// CB indices
constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_in = 1;
constexpr uint32_t cb_x_sq = 2;
constexpr uint32_t cb_scaler = 8;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_out_rm = 17;

// Compile-time args
constexpr uint32_t is_rm_input = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);
constexpr uint32_t Ht = get_compile_time_arg_val(3);
constexpr uint32_t NC = get_compile_time_arg_val(4);

void kernel_main() {
    uint32_t num_rows = get_arg_val<uint32_t>(0);

    // Hardware init: srcA=cb_in, srcB=cb_scaler, out=cb_out
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Phase 1: Tilize (RM input only)
        if constexpr (is_rm_input) {
            compute_kernel_lib::tilize<
                Wt,
                cb_in_rm,
                cb_in,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(1);
        }

        // Phase 2: Square (cb_in -> cb_x_sq)
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_in, cb_x_sq, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 3: Reduce Row (cb_x_sq -> cb_out)
        // Uses SUM with scaler 1/W to compute mean(x^2)
        // Output goes directly to cb_out (1 tile per row)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_x_sq, cb_scaler, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt, 1));

        // Phase 8: Untilize (RM output only)
        if constexpr (is_rm_input) {
            compute_kernel_lib::untilize<
                1,
                cb_out,
                cb_out_rm,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
        }
    }
}
