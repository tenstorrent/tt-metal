// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Unified compute kernel for toy_reduce_partial.
//
// Handles both REDUCE_ROW (W) and REDUCE_COL (H) via REDUCE_ROW_MODE
// compile-time arg. Supports MAX and SUM pool types via POOL_TYPE_SUM arg.
// The scaler tiles are owned by this kernel via ReduceScaler::compute_managed():
// it emits a partial scaler for the last tile in the reduced dimension (last W
// tile for REDUCE_ROW, last H tile for REDUCE_COL) when PARTIAL_DIM is set.

#include <cstdint>

#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t partial_dim = get_compile_time_arg_val(3);
    constexpr uint32_t reduce_row_mode = get_compile_time_arg_val(4);
    constexpr uint32_t pool_type_sum = get_compile_time_arg_val(5);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    constexpr auto partial_scaler = compute_kernel_lib::ReduceScaler::compute_managed(partial_dim);

    constexpr auto block_shape = compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC);

    if constexpr (pool_type_sum) {
        if constexpr (reduce_row_mode) {
            compute_kernel_lib::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_in,
                cb_scaler,
                cb_out,
                compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                block_shape,
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                compute_kernel_lib::NoOp{},
                partial_scaler);
        } else {
            compute_kernel_lib::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_COL,
                cb_in,
                cb_scaler,
                cb_out,
                compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                block_shape,
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                compute_kernel_lib::NoOp{},
                partial_scaler);
        }
    } else {
        if constexpr (reduce_row_mode) {
            compute_kernel_lib::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                cb_in,
                cb_scaler,
                cb_out,
                compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                block_shape,
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                compute_kernel_lib::NoOp{},
                partial_scaler);
        } else {
            compute_kernel_lib::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_COL,
                cb_in,
                cb_scaler,
                cb_out,
                compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                block_shape,
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                compute_kernel_lib::NoOp{},
                partial_scaler);
        }
    }
}
