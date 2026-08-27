// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file numeric.h
 * @brief Generic numeric/math utilities for compute kernels.
 *
 * The row-wise reduction helpers here are thin adapters over
 * compute_kernel_lib::reduce<>. They exist to keep the normalization kernels' vocabulary
 * (num_tiles / block_size / N / a policy struct) while the reduction itself, the partial-scaler
 * handling and the DST lifecycle all live in the shared reduce helper.
 *
 * Because reduce<> takes its circular-buffer ids as *template* parameters, these adapters do too;
 * callers pass ids rather than DataflowBuffer objects.
 */

#pragma once

#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/operations/normalization/kernel_util/compute/policies.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/kernel_util/generic/bit.h"
#include <type_traits>

namespace policies = norm::kernel_util::compute::policies;
namespace generic = norm::kernel_util::generic;

namespace norm::kernel_util::compute::numeric {

namespace detail {

constexpr uint32_t dst0 = 0;

/**
 * @brief Convenience no-op epilogue. Takes the DST index, matching reduce<>'s post_reduce_op.
 */
constexpr auto no_op = [](uint32_t) {};

/**
 * @brief Scale the destination register tile data by a scalar
 */
inline void scale_dest(uint32_t dst, uint32_t scalar) {
    binop_with_scalar_tile_init();
    mul_unary_tile(dst, scalar);
}

/**
 * @brief Map a normalization input policy onto a reduce<> input policy.
 *
 * `pop` streams one tile at a time. That is the only safe choice here: the large-tensor kernel
 * deliberately sizes its input CB below the reduce extent, so any bulk wait would deadlock.
 * `!pop` leaves the tiles resident and indexes them, which is what WaitUpfrontNoPop does.
 *
 * Note this trades the caller's block_size-granular CB handshake for a per-tile one. The block
 * granularity is not expressible with the stock reduce<> policies.
 */
template <typename input_policy>
constexpr compute_kernel_lib::ReduceInputPolicy to_reduce_policy() {
    return input_policy::pop ? compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile
                             : compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop;
}

/**
 * @brief Number of tiles the producer pushed beyond `num_tiles` because it pads to whole blocks.
 *
 * The normalization producers reserve and push `full_block_size()` for the final short block while
 * only filling the real tiles. reduce<> pops exactly what it reduces, so the padding has to be
 * drained separately or the CB desynchronises across the enclosing NCHt loop.
 */
inline uint32_t block_padding(uint32_t num_tiles, uint32_t block_size) {
    return generic::blocks(num_tiles, block_size).total_with_remainder() - num_tiles;
}

template <typename input_policy>
inline void drain_block_padding(uint32_t dfb_id, uint32_t num_tiles, uint32_t block_size) {
    if constexpr (input_policy::pop && input_policy::sync_full_block) {
        const uint32_t pad = block_padding(num_tiles, block_size);
        if (pad > 0) {
            DataflowBuffer dfb(dfb_id);
            dfb.wait_front(pad);
            dfb.pop_front(pad);
        }
    }
}

/**
 * @brief The partial-scaler selector for a reduce of `N` elements over `tile_width`-wide tiles.
 *
 * The reader emits a second, partial-fill scaler tile when the last tile along the reduce dimension
 * is ragged; reduce<> applies it to that tile so the padding columns contribute nothing.
 */
inline compute_kernel_lib::ReducePartialScaler partial_scaler_for(uint32_t N, uint32_t tile_width) {
    return (N % tile_width > 0) ? compute_kernel_lib::ReducePartialScaler::last_tile()
                                : compute_kernel_lib::ReducePartialScaler::none();
}

}  // namespace detail

/**
 * @brief Reduce along the rows of tiles in a CB, apply an optional epilogue in DST, and push the
 * single result tile to an output CB.
 *
 * @tparam reduce_type Pool type (SUM / AVG / MAX)
 * @tparam reduce_dim Reduction dimension
 * @tparam in_dfb_id Input CB id
 * @tparam scalar_dfb_id Scaler CB id. See \ref scalar_tile_ones, \ref partial_tile_scaler_tiles
 * @tparam out_dfb_id Output CB id (one tile is pushed)
 * @tparam input_policy How to consume the input CB
 * @tparam wait_at_end_policy Whether to wait on the output tile before returning
 * @tparam Epilogue Callable taking the DST index, run on the accumulator before packing
 *
 * @param num_tiles Number of tiles along the reduce dimension
 * @param block_size The producer's block granularity; used only to drain its block padding
 * @param N Number of real elements reduced (drives the partial-scaler selection)
 * @param tile_width Tile extent along the reduce dimension
 * @param epilogue Runs once on the accumulator
 *
 * @note dst0 is used to accumulate, so it will be overwritten @anchor dst0_overwritten
 * @note It is up to the caller to ensure the scalar tile is correctly populated. If it doesn't
 * contain 1's, the result will be incorrect @anchor scalar_tile_ones
 * @note If the last tile is partial, the scaler CB needs two tiles: one for the full tiles and one
 * for the partial tile @anchor partial_tile_scaler_tiles
 */
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    uint32_t in_dfb_id,
    uint32_t scalar_dfb_id,
    uint32_t out_dfb_id,
    typename input_policy = policies::PartialBlockWithoutPopPolicy,
    policies::WaitAtEndPolicy wait_at_end_policy = policies::WaitAtEndPolicy::WAIT,
    typename Epilogue = decltype(detail::no_op)>
inline void row_wise_accumulate_with_epilogue(
    uint32_t num_tiles, uint32_t block_size, uint32_t N, uint32_t tile_width = 32, Epilogue epilogue = detail::no_op) {
    compute_kernel_lib::
        reduce<reduce_type, reduce_dim, in_dfb_id, scalar_dfb_id, out_dfb_id, detail::to_reduce_policy<input_policy>()>(
            compute_kernel_lib::ReduceInputBlockShape::row(num_tiles),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            epilogue,
            detail::partial_scaler_for(N, tile_width));

    detail::drain_block_padding<input_policy>(in_dfb_id, num_tiles, block_size);

    if constexpr (wait_at_end_policy == policies::WaitAtEndPolicy::WAIT) {
        DataflowBuffer(out_dfb_id).wait_front(1);
    }
}

/**
 * @brief Compute the row-wise mean of an entire input CB
 *
 * See \ref dst0_overwritten, \ref scalar_tile_ones, \ref partial_tile_scaler_tiles
 */
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    uint32_t in_dfb_id,
    uint32_t scalar_dfb_id,
    uint32_t out_dfb_id,
    typename input_policy = policies::PartialBlockWithoutPopPolicy,
    policies::WaitAtEndPolicy wait_at_end_policy = policies::WaitAtEndPolicy::WAIT>
inline void row_wise_mean(uint32_t N, uint32_t num_tiles, uint32_t block_size, uint32_t tile_width = 32) {
    row_wise_accumulate_with_epilogue<
        reduce_type,
        reduce_dim,
        in_dfb_id,
        scalar_dfb_id,
        out_dfb_id,
        input_policy,
        wait_at_end_policy>(num_tiles, block_size, N, tile_width, [N](uint32_t dst) {
        detail::scale_dest(dst, generic::bit_cast<uint32_t>(1.0f / N));
    });
}

/**
 * @brief Compute the row-wise mean of the elementwise sum of two input CBs.
 *
 * The elementwise sum is never materialised. By linearity,
 *
 *     E[x + b] = (1/N) * sum(x_i + b_i) = (1/N) * (sum(x_i) + sum(b_i))
 *
 * so this reduces the first CB, then folds the second one in on top of the running result before
 * dividing. The fold uses reduce<>'s Accumulate, which reloads the first pass's packed tile from
 * `out_dfb_id` — so the intermediate makes one round trip through that CB's data format.
 *
 * See \ref dst0_overwritten, \ref scalar_tile_ones, \ref partial_tile_scaler_tiles
 */
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    uint32_t in0_dfb_id,
    uint32_t in1_dfb_id,
    uint32_t scalar_dfb_id,
    uint32_t out_dfb_id,
    typename input_policy = policies::PartialBlockWithoutPopPolicy,
    policies::WaitAtEndPolicy wait_at_end_policy = policies::WaitAtEndPolicy::WAIT>
inline void row_wise_mean_with_pre_add(uint32_t N, uint32_t num_tiles, uint32_t block_size, uint32_t tile_width = 32) {
    constexpr auto policy = detail::to_reduce_policy<input_policy>();
    const auto partial = detail::partial_scaler_for(N, tile_width);
    const auto shape = compute_kernel_lib::ReduceInputBlockShape::row(num_tiles);
    const auto layout = compute_kernel_lib::ReduceInputMemoryLayout::contiguous();

    // Pass 1: sum(in0) -> out. Preserve the unfinalized partial so pass 2 can fold in sum(in1);
    // the division and reduction finalization happen once, after both passes.
    compute_kernel_lib::reduce<reduce_type, reduce_dim, in0_dfb_id, scalar_dfb_id, out_dfb_id, policy>(
        shape, layout, compute_kernel_lib::Accumulate::at(out_dfb_id, /*iteration=*/0), detail::no_op, partial);
    detail::drain_block_padding<input_policy>(in0_dfb_id, num_tiles, block_size);

    // Pass 2: sum(in1) folded onto the reloaded pass-1 result, then divide by N.
    compute_kernel_lib::reduce<reduce_type, reduce_dim, in1_dfb_id, scalar_dfb_id, out_dfb_id, policy>(
        shape,
        layout,
        compute_kernel_lib::Accumulate::at_last(out_dfb_id, /*iteration=*/1),
        [N](uint32_t dst) { detail::scale_dest(dst, generic::bit_cast<uint32_t>(1.0f / N)); },
        partial);
    detail::drain_block_padding<input_policy>(in1_dfb_id, num_tiles, block_size);

    if constexpr (wait_at_end_policy == policies::WaitAtEndPolicy::WAIT) {
        DataflowBuffer(out_dfb_id).wait_front(1);
    }
}

}  // namespace norm::kernel_util::compute::numeric
