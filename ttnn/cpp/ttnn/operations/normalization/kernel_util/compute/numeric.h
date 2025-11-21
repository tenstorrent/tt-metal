// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file numeric.h
 * @brief Generic numeric/math utilities for compute kernels.
 */

#pragma once

#include "ttnn/operations/normalization/kernel_util/generic/policies.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"

namespace policies = norm::kernel_util::generic::policies;

namespace norm::kernel_util::compute::numeric {

/**
 * @brief Compute the row-wise mean of an entire input CB.
 * @param cb_in The input CB to compute mean of
 * @param cb_scalar The CB containing a scalar reduce tile of 1's
 * @param cb_out The output CB to store the mean in
 * @param one_over_N The inverse of the number of entries in the
 * mean (1/N) encoded as a uint32_t
 * @param num_tiles The number of tiles in the mean
 * before processing the next block. Must be an even number
 * @param block_size The number of tiles to wait for at a time
 * @tparam FLOAT32_REDUCTION Whether to reduce the sum in FP32 precision
 * @tparam pop_input_policy The policy for whether to pop the input CB after processing
 * @tparam wait_at_end_policy The policy for whether to wait at the end of the function
 *
 * @note dst0 is used to accumuate the sum, so it
 * will be overwritten here
 * @note It is up to the caller to ensure that the scalar tile
 * is correctly populated. If it doesn't contain 1's, the result
 * will be incorrect
 * @note This function must be called from a compilation unit
 * that has the following preprocessor directives (this is for
 * the reduce interface):
 * #define REDUCE_OP PoolType::SUM
 * #define REDUCE_DIM ReduceDim::REDUCE_ROW
 */
template <
    bool FLOAT32_REDUCTION,
    policies::PopInputPolicy pop_input_policy = policies::PopInputPolicy::NO_POP,
    policies::WaitAtEndPolicy wait_at_end_policy = policies::WaitAtEndPolicy::WAIT>
inline void row_wise_mean(
    uint32_t cb_in, uint32_t cb_scalar, uint32_t cb_out, uint32_t one_over_N, uint32_t num_tiles, uint32_t block_size) {
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler_tile_idx = 0;
    constexpr bool pop_input = pop_input_policy == policies::PopInputPolicy::POP;
    constexpr bool wait_at_end = wait_at_end_policy == policies::WaitAtEndPolicy::WAIT;

    reconfig_data_format(cb_in, cb_scalar);
    tile_regs_acquire();
    reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb_in, cb_scalar, cb_out);
    for (uint32_t t = 0; t < num_tiles; t += block_size) {
        const uint32_t num_previous_tiles = pop_input ? 0 : t;
        cb_wait_front(cb_in, num_previous_tiles + block_size);
        for (uint32_t j = 0; j < block_size; j++) {
            reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(
                cb_in, cb_scalar, num_previous_tiles + j, scaler_tile_idx, dst0);
        }
        if constexpr (pop_input) {
            cb_pop_front(cb_in, block_size);
        }
    }
    reduce_uninit<FLOAT32_REDUCTION>();

    // Scale the accumulated sum
    binop_with_scalar_tile_init();
    mul_unary_tile(dst0, one_over_N);

    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_out, 1);
    pack_reconfig_data_format(cb_out);
    pack_tile(dst0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);

    if constexpr (wait_at_end) {
        cb_wait_front(cb_out, 1);
    }
}
}  // namespace norm::kernel_util::compute::numeric
