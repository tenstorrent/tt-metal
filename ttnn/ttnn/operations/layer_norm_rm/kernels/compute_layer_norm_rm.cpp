// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 2 (reduce_mean): tilize c_0->c_1, reduce_row c_1->c_24 (mean), untilize c_24->c_17

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

constexpr uint32_t c_0 = 0;    // RM input staging for tilize
constexpr uint32_t c_1 = 1;    // Tilized input tiles
constexpr uint32_t c_2 = 2;    // Reduce scaler 1/W
constexpr uint32_t c_24 = 24;  // Row-wise mean (1 tile)
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

        // Phase 2: Reduce row -> mean
        // reduce<SUM, REDUCE_ROW> with WaitAndPopPerTile (streaming), INPUT_AND_OUTPUT reconfig
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            c_1, c_2, c_24, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 13: Untilize c_24 -> c_17 (reduced: 1 tile per tile-row)
        compute_kernel_lib::untilize<
            1,
            c_24,
            c_17,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
    }
}
