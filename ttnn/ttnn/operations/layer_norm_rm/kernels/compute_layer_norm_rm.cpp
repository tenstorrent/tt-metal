// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 2: tilize -> reduce_mean -> untilize (row-wise mean output)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"
// CB indices
constexpr uint32_t cb_input_rm = 0;       // c_0: RM sticks from reader
constexpr uint32_t cb_tilized = 1;        // c_1: Tilized tiles
constexpr uint32_t cb_reduce_scaler = 2;  // c_2: Reduce scaler (1/W)
constexpr uint32_t cb_mean = 24;          // c_24: Row-reduced mean (1 tile)
constexpr uint32_t cb_output_tiles = 16;  // c_16: Pre-untilize output tiles
constexpr uint32_t cb_output_rm = 17;     // c_17: Untilized RM output

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);

void kernel_main() {
    // Runtime args
    const uint32_t nblocks = get_arg_val<uint32_t>(0);

    if (nblocks == 0) {
        return;
    }

    // Hardware startup: input CB for srcA/srcB, output CB for pack
    compute_kernel_hw_startup(cb_input_rm, cb_reduce_scaler, cb_output_rm);

    for (uint32_t block = 0; block < nblocks; block++) {
        // Phase 1: Tilize c_0 -> c_1
        compute_kernel_lib::tilize<
            cb_input_rm,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // Phase 2: Reduce mean - SUM along row with 1/W scaler
        // Input: c_1 (Wt tiles), Scaler: c_2 (1 tile, 1/W), Output: c_24 (1 tile, col vector)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Untilize mean (1 tile wide) from c_24 -> c_17
        compute_kernel_lib::untilize<
            1,
            cb_mean,
            cb_output_rm,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
    }
}
