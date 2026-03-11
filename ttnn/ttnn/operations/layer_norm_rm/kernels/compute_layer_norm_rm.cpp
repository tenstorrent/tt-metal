// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 4 (variance): Pass 1 (tilize+reduce->mean), Pass 2 (tilize+sub+square+reduce->variance)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

constexpr uint32_t c_0 = 0;    // RM input staging for tilize
constexpr uint32_t c_1 = 1;    // Tilized input tiles
constexpr uint32_t c_2 = 2;    // Reduce scaler 1/W
constexpr uint32_t c_24 = 24;  // Row-wise mean (1 tile)
constexpr uint32_t c_25 = 25;  // Intermediate tiles
constexpr uint32_t c_26 = 26;  // Row-wise variance (1 tile)
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
        // === Pass 1: Tilize + Reduce -> Mean ===

        // Phase 1: Tilize c_0 -> c_1
        compute_kernel_lib::tilize<
            c_0,
            c_1,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(Wt, 1);

        // Phase 2: Reduce row -> mean (c_1 -> c_24)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            c_1, c_2, c_24, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // === Pass 2: Tilize + Subtract Mean ===

        // Phase 3: Tilize c_0 -> c_1 (pass 2)
        compute_kernel_lib::tilize<
            c_0,
            c_1,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(Wt, 1);

        // Phase 4: Sub mean -> centered: sub<COL>(c_1, c_24) -> c_25
        // A: c_1 (Wt tiles, popped per tile)
        // B: c_24 (1 tile mean, waited upfront, NOT popped -- persists for later)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            c_1, c_24, c_25, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5: Square centered: square(c_25) -> c_1
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            c_25, c_1, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 6: Reduce row -> variance: reduce<SUM,REDUCE_ROW>(c_1) -> c_26
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            c_1, c_2, c_26, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Untilize c_26 (1 tile, variance) -> c_17
        compute_kernel_lib::untilize<
            1,
            c_26,
            c_17,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);

        // End of tile-row: pop persistent per-tile-row CBs
        cb_pop_front(c_24, 1);  // mean
    }
}
