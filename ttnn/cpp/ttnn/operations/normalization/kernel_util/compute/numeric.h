// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file numeric.h
 * @brief Generic numeric/math utilities for compute kernels.
 */

#pragma once

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "ttnn/operations/normalization/kernel_util/compute/policies.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/kernel_util/generic/bit.h"
#include <tt-metalium/constants.hpp>
#include <type_traits>
#include <array>

namespace policies = norm::kernel_util::compute::policies;
namespace generic = norm::kernel_util::generic;

namespace norm::kernel_util::compute::numeric {

namespace detail {

constexpr uint32_t dst0 = 0;

/**
 * @brief Convenience no-op function
 */
constexpr auto no_op = []() {};

/**
 * @brief Scale the destination register tile data by a scalar
 *
 * @param dst The destination tile to scale
 * @param scalar The scalar to scale the destination tile by
 */
inline void scale_dest(uint32_t dst, uint32_t scalar) {
    binop_with_scalar_tile_init();
    mul_unary_tile(dst, scalar);
}

/**
 * @brief The compute logic for accumulating a CB. Does
 * no src configuring or packing
 */
template <bool FLOAT32_REDUCTION, typename input_policy, typename... AdditionalCBs>
inline void accumulate_compute_loop(
    uint32_t cb_in,
    uint32_t cb_scalar,
    uint32_t cb_out,
    uint32_t num_tiles,
    uint32_t block_size,
    bool last_tile_partial,
    AdditionalCBs... cb_additional) {
    static_assert(
        (std::conjunction_v<std::is_same<AdditionalCBs, uint32_t>...>), "All additional CBs must be uint32_t");

    constexpr bool pop_input = input_policy::pop;
    constexpr bool sync_full_block = input_policy::sync_full_block;

    auto accumulate_cb = [cb_scalar, block_size, cb_out, num_tiles, last_tile_partial](uint32_t cb) {
        reduce_init<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(cb, cb_scalar, cb_out);
        for (auto block : generic::blocks(num_tiles, block_size)) {
            const auto num_previous_tiles = pop_input ? 0 : block.start();
            const auto curr_block_size = sync_full_block ? block.full_block_size() : block.size();
            const uint32_t num_tiles_to_wait = num_previous_tiles + curr_block_size;
            cb_wait_front(cb, num_tiles_to_wait);
            for (auto j : block.local()) {
                // If it's the last tile and it's partial, use the second tile in cb_scalar
                const auto scaler_tile_idx = block.to_global(j) == num_tiles - 1 && last_tile_partial ? 1 : 0;
                reduce_tile<REDUCE_OP, REDUCE_DIM, FLOAT32_REDUCTION>(
                    cb, cb_scalar, num_previous_tiles + j, scaler_tile_idx, detail::dst0);
            }
            if constexpr (pop_input) {
                cb_pop_front(cb, curr_block_size);
            }
        }
    };

    // Accumulate the input CB
    accumulate_cb(cb_in);

    // Accumulate any additional CBs
    constexpr uint32_t num_additional_cbs = sizeof...(cb_additional);
    if constexpr (num_additional_cbs > 0) {
        const std::array<uint32_t, num_additional_cbs> additional_cbs_array = {cb_additional...};

        for (uint32_t i = 0; i < num_additional_cbs; i++) {
            accumulate_cb(additional_cbs_array[i]);
        }
    }

    reduce_uninit<FLOAT32_REDUCTION>();
}

}  // namespace detail

/**
 * @brief Accumulate (sum) along the rows of tiles in a CB, apply
 * an optional compute epilogue, and push the result to an output CB
 *
 * @param cb_in Input CB to accumulate
 * @param cb_scalar CB containing the scalar tile to use in reduce
 * @param cb_out Output CB to store the accumulated value
 * @param num_tiles Number of tiles containing the data
 * @param block_size Number of tiles to process at a time
 * @param N Number of entries in the set
 * @param epilogue Optional functor to call after the accumulation before tile registers
 * are committed and packed
 * @param additional_cbs Optional additional input CBs to accumulate
 * @tparam FLOAT32_REDUCTION Whether to reduce the sum in FP32 precision
 * @tparam input_policy The policy for how to handle the input CB
 * @tparam wait_at_end_policy The policy for whether to wait at the end of the function
 * @tparam Epilogue The type of the epilogue functor
 * @tparam AdditionalCBs The types of the additional input CBs (must be uint32_t)
 *
 * @note dst0 is used to accumuate the sum, so it
 * will be overwritten here @anchor dst0_overwritten
 * @note It is up to the caller to ensure that the scalar tile
 * is correctly populated. If it doesn't contain 1's, the result
 * will be incorrect @anchor scalar_tile_ones
 * @note If the last tile is partial, we need two scalar tiles in cb_scalar:
 * One for the full tiles, and one for the partial tile @anchor partial_tile_scaler_tiles
 * @note This function must be called from a compilation unit
 * that has the following preprocessor directives (this is for
 * the reduce interface):
 * #define REDUCE_OP PoolType::SUM
 * #define REDUCE_DIM ReduceDim::REDUCE_ROW @anchor reduce_defines
 * @note All input CBs will wait on the same number of tiles in a block,
 * and will have the same pop policy
 * @note This first streams all `num_tiles` tiles from `cb_in0` then
 * streams all `num_tiles` tiles from `cb_in1` @anchor stream_cbs
 */
template <
    bool FLOAT32_REDUCTION,
    typename input_policy = policies::PartialBlockWithoutPopPolicy,
    policies::WaitAtEndPolicy wait_at_end_policy = policies::WaitAtEndPolicy::WAIT,
    typename Epilogue = decltype(detail::no_op),
    typename... AdditionalCBs>
inline void row_wise_accumulate_with_epilogue(
    uint32_t cb_in,
    uint32_t cb_scalar,
    uint32_t cb_out,
    uint32_t num_tiles,
    uint32_t block_size,
    uint32_t N,
    Epilogue epilogue = detail::no_op,
    AdditionalCBs... additional_cbs) {
    constexpr bool wait_at_end = wait_at_end_policy == policies::WaitAtEndPolicy::WAIT;
    // If the last tile is partial, we need two scalar tiles:
    // One for the full tiles, and one for the partial tile
    const auto last_tile_partial = N % tt::constants::TILE_WIDTH > 0;
    const uint32_t num_scaler_tiles_needed = last_tile_partial ? 2 : 1;
    cb_wait_front(cb_scalar, num_scaler_tiles_needed);
    reconfig_data_format(cb_in, cb_scalar);
    tile_regs_acquire();

    detail::accumulate_compute_loop<FLOAT32_REDUCTION, input_policy>(
        cb_in, cb_scalar, cb_out, num_tiles, block_size, last_tile_partial, additional_cbs...);

    epilogue();

    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_out, 1);
    pack_reconfig_data_format(cb_out);
    pack_tile(detail::dst0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);

    if constexpr (wait_at_end) {
        cb_wait_front(cb_out, 1);
    }
}

/**
 * @brief Compute the row-wise mean of an entire input CB
 *
 * @param cb_in Input CB to compute mean of
 * @param cb_scalar CB containing a scalar reduce tile of 1's.
 * See \ref partial_tile_scaler_tiles for additional requirements
 * @param cb_out Output CB to store the mean
 * @param N Number of entries in the set
 * @param num_tiles Number of tiles containing the data
 * @param block_size Number of tiles to process at a time
 * @tparam FLOAT32_REDUCTION Whether to reduce the sum in FP32 precision
 * @tparam input_policy The policy for how to handle the input CB
 * @tparam wait_at_end_policy The policy for whether to wait at the end of the function
 *
 * See \ref dst0_overwritten, \ref scalar_tile_ones, \ref reduce_defines, \ref stream_cbs, \ref
 * partial_tile_scaler_tiles
 */
template <
    bool FLOAT32_REDUCTION,
    typename input_policy = policies::PartialBlockWithoutPopPolicy,
    policies::WaitAtEndPolicy wait_at_end_policy = policies::WaitAtEndPolicy::WAIT>
inline void row_wise_mean(
    uint32_t cb_in, uint32_t cb_scalar, uint32_t cb_out, uint32_t N, uint32_t num_tiles, uint32_t block_size) {
    row_wise_accumulate_with_epilogue<FLOAT32_REDUCTION, input_policy, wait_at_end_policy>(
        cb_in, cb_scalar, cb_out, num_tiles, block_size, N, [N]() {
            detail::scale_dest(detail::dst0, generic::bit_cast<uint32_t>(1.0f / N));
        });
}

/**
 * @brief Compute the row-wise mean of the sum of two input CBs
 *
 * @param cb_in Input CB to compute mean of
 * @param cb_in1 Additional input CB to accumulate
 * @param cb_scalar CB containing a scalar reduce tile of 1's.
 * See \ref partial_tile_scaler_tiles for additional requirements
 * @param cb_out Output CB to store the mean
 * @param N Number of entries in the set
 * @param num_tiles Number of tiles containing the data
 * @param block_size Number of tiles to process at a time
 * @tparam FLOAT32_REDUCTION Whether to reduce the sum in FP32 precision
 * @tparam input_policy The policy for how to handle the input CB
 * @tparam wait_at_end_policy The policy for whether to wait at the end of the function
 *
 * See \ref dst0_overwritten, \ref scalar_tile_ones, \ref reduce_defines, \ref stream_cbs, \ref
 * partial_tile_scaler_tiles
 */
template <
    bool FLOAT32_REDUCTION,
    typename input_policy = policies::PartialBlockWithoutPopPolicy,
    policies::WaitAtEndPolicy wait_at_end_policy = policies::WaitAtEndPolicy::WAIT>
inline void row_wise_mean_with_pre_add(
    uint32_t cb_in0,
    uint32_t cb_in1,
    uint32_t cb_scalar,
    uint32_t cb_out,
    uint32_t N,
    uint32_t num_tiles,
    uint32_t block_size) {
    row_wise_accumulate_with_epilogue<FLOAT32_REDUCTION, input_policy, wait_at_end_policy>(
        cb_in0,
        cb_scalar,
        cb_out,
        num_tiles,
        block_size,
        N,
        [N]() { detail::scale_dest(detail::dst0, generic::bit_cast<uint32_t>(1.0f / N)); },
        cb_in1);
}
}  // namespace norm::kernel_util::compute::numeric
