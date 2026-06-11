// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for streaming_reduce_helpers.hpp
// Do not include directly — include streaming_reduce_helpers.hpp instead.

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace compute_kernel_lib {

template <
    PoolType pool,
    ReduceDim rdim,
    ReduceInputPolicy in_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    typename PostOp>
ALWI void accumulate_reduce_block(
    uint32_t cb_in,
    uint32_t cb_scaler,
    uint32_t cb_acc,
    ReduceInputBlockShape block_shape,
    uint32_t b,
    uint32_t num_blocks,
    ReducePartialScaler partial,
    PostOp post_op_final) {
    const bool is_last = (b + 1 == num_blocks);
    if (is_last) {
        reduce<pool, rdim, in_policy, reconfig_mode>(
            cb_in,
            cb_scaler,
            cb_acc,
            block_shape,
            ReduceInputMemoryLayout::contiguous(),
            Accumulate::at(cb_acc, b),
            post_op_final,
            partial);
    } else {
        reduce<pool, rdim, in_policy, reconfig_mode>(
            cb_in,
            cb_scaler,
            cb_acc,
            block_shape,
            ReduceInputMemoryLayout::contiguous(),
            Accumulate::at(cb_acc, b),
            NoOp{},
            ReducePartialScaler::none());
    }
}

template <
    PoolType pool,
    ReduceDim rdim,
    ReduceInputPolicy in_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    typename PostOp>
ALWI void accumulate_reduce(
    uint32_t cb_in,
    uint32_t cb_scaler,
    uint32_t cb_acc,
    ReduceInputBlockShape block_shape,
    uint32_t num_blocks,
    ReducePartialScaler partial,
    PostOp post_op_final) {
    for (uint32_t b = 0; b < num_blocks; ++b) {
        accumulate_reduce_block<pool, rdim, in_policy, reconfig_mode>(
            cb_in, cb_scaler, cb_acc, block_shape, b, num_blocks, partial, post_op_final);
    }
}

}  // namespace compute_kernel_lib
