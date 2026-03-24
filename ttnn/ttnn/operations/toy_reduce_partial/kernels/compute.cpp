// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Unified compute kernel for toy_reduce_partial.
//
// Handles both REDUCE_ROW (W) and REDUCE_COL (H) via REDUCE_ROW_MODE
// compile-time arg. The partial scaler mechanism works identically for both:
// the reduce helper selects scaler tile 1 for the last tile in the reduced
// dimension (last W tile for REDUCE_ROW, last H tile for REDUCE_COL).

#include <cstdint>

#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t has_partial = get_compile_time_arg_val(3);
    constexpr uint32_t reduce_row_mode = get_compile_time_arg_val(4);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    constexpr auto partial_scaler = has_partial ? compute_kernel_lib::ReducePartialScaler::last_tile_at(1)
                                                : compute_kernel_lib::ReducePartialScaler::none();

    constexpr auto block_shape = compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC);

    if constexpr (reduce_row_mode) {
        compute_kernel_lib::reduce<
            PoolType::MAX,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
            cb_in,
            cb_scaler,
            cb_out,
            block_shape,
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            compute_kernel_lib::NoOp{},
            partial_scaler);
    } else {
        compute_kernel_lib::reduce<
            PoolType::MAX,
            ReduceDim::REDUCE_COL,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
            cb_in,
            cb_scaler,
            cb_out,
            block_shape,
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            compute_kernel_lib::NoOp{},
            partial_scaler);
    }

    constexpr uint32_t num_scaler_tiles = has_partial ? 2 : 1;
    cb_pop_front(cb_scaler, num_scaler_tiles);
}
