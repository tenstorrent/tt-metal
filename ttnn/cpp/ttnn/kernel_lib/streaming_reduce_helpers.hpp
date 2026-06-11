// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

/**
 * @file streaming_reduce_helpers.hpp
 * @brief Streaming-reduce primitives that wrap reduce<> + Accumulate::at and
 *        a small in-place 1-tile transform helper.
 *
 * Motivation: when the reduction axis does not fit in L1, callers must run a
 * sequence of accumulating reduce<> calls with the right partial-scaler /
 * post-op routing applied only to the last block. These helpers bundle that
 * routing so callers do not re-derive it per op.
 *
 * The helpers do not own format reconfig across blocks beyond what the
 * underlying reduce<> already does. They are wrappers, not new primitives.
 *
 * Primitives:
 *  - accumulate_reduce<>     : full streaming reduce — owns the block loop
 *  - accumulate_reduce_block<> : single-iteration helper for callers that
 *    interleave per-block work between sub/mul and reduce
 *
 * For an in-place SFPU finalizer on the accumulator (e.g. rsqrt-with-eps), use
 * `compute_kernel_lib::transform_in_place` from eltwise_convenience.hpp.
 */

namespace compute_kernel_lib {

// =============================================================================
// accumulate_reduce_block — single-block streaming reduce with last-block routing
// =============================================================================
//
// Performs ONE iteration of an accumulating reduce. The helper:
//   - always uses Accumulate::at(cb_acc, b) — index-aware reload
//   - if b == num_blocks - 1: forwards `partial` and `post_op_final` to reduce<>
//   - otherwise: forwards ReducePartialScaler::none() and NoOp{}
//
// Use this when you need to interleave per-block work (e.g. per-block sub/mul
// before the reduce). Use `accumulate_reduce` if you don't.
//
// Caller owns:
//   - sizing cb_acc to (block_shape.rows * block_shape.batches) pages
//   - popping cb_acc after the call chain when consumption is done
//   - ensuring cb_in has tiles available per `in_policy`
template <
    PoolType pool,
    ReduceDim rdim,
    ReduceInputPolicy in_policy = ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode reconfig_mode = ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
    typename PostOp = NoOp>
ALWI void accumulate_reduce_block(
    uint32_t cb_in,
    uint32_t cb_scaler,
    uint32_t cb_acc,
    ReduceInputBlockShape block_shape,
    uint32_t b,
    uint32_t num_blocks,
    ReducePartialScaler partial = ReducePartialScaler::none(),
    PostOp post_op_final = PostOp{});

// =============================================================================
// accumulate_reduce — full streaming reduce (owns the block loop)
// =============================================================================
//
// Streams `num_blocks` blocks of `block_shape` from cb_in into cb_acc, calling
// reduce<> with Accumulate::at(cb_acc, b) for each block. After the call
// returns, cb_acc holds the final reduced value(s).
//
// Partial-scaler routing: applied ONLY to the last block. All earlier blocks
// run with ReducePartialScaler::none(). The caller passes the partial scaler
// as if for the whole streaming reduction; the helper takes care of routing.
//
// Post-op routing: `post_op_final` runs only on the last block, via reduce<>'s
// post_reduce_op hook (in DST after final accumulation, before pack). For
// multi-instruction finalizers (e.g. rsqrt-with-eps), pass NoOp{} here and run
// `compute_kernel_lib::transform_in_place` (eltwise_convenience.hpp) on cb_acc
// afterwards.
template <
    PoolType pool,
    ReduceDim rdim,
    ReduceInputPolicy in_policy = ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode reconfig_mode = ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
    typename PostOp = NoOp>
ALWI void accumulate_reduce(
    uint32_t cb_in,
    uint32_t cb_scaler,
    uint32_t cb_acc,
    ReduceInputBlockShape block_shape,
    uint32_t num_blocks,
    ReducePartialScaler partial = ReducePartialScaler::none(),
    PostOp post_op_final = PostOp{});

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.inl"
