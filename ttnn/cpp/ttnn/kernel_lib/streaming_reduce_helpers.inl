// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for streaming_reduce_helpers.hpp
// Do not include directly — include streaming_reduce_helpers.hpp instead.

#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace compute_kernel_lib {

template <
    PoolType pool,
    ReduceDim rdim,
    uint32_t cb_in,
    uint32_t cb_scaler,
    uint32_t cb_acc,
    ReduceInputPolicy in_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    typename PostOp>
ALWI void accumulate_reduce_block(
    ReduceInputBlockShape block_shape,
    uint32_t b,
    uint32_t num_blocks,
    ReducePartialScaler partial,
    PostOp post_op_final) {
    const bool is_last = (b + 1 == num_blocks);
    if (is_last) {
        reduce<pool, rdim, cb_in, cb_scaler, cb_acc, in_policy, reconfig_mode>(
            block_shape,
            ReduceInputMemoryLayout::contiguous(),
            Accumulate::at(cb_acc, b),
            post_op_final,
            partial);
    } else {
        reduce<pool, rdim, cb_in, cb_scaler, cb_acc, in_policy, reconfig_mode>(
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
    uint32_t cb_in,
    uint32_t cb_scaler,
    uint32_t cb_acc,
    ReduceInputPolicy in_policy,
    ReduceDataFormatReconfigMode reconfig_mode,
    typename PostOp>
ALWI void accumulate_reduce(
    ReduceInputBlockShape block_shape,
    uint32_t num_blocks,
    ReducePartialScaler partial,
    PostOp post_op_final) {
    for (uint32_t b = 0; b < num_blocks; ++b) {
        accumulate_reduce_block<pool, rdim, cb_in, cb_scaler, cb_acc, in_policy, reconfig_mode>(
            block_shape, b, num_blocks, partial, post_op_final);
    }
}

template <typename Transform>
ALWI void transform_in_place(uint32_t cb, Transform t) {
    constexpr uint32_t onetile = 1;

    cb_wait_front(cb, onetile);

    tile_regs_acquire();
    // Bundled format reconfig: SRCA <- cb, packer <- cb. Both happen
    // unconditionally so a previous phase that left SRCA / packer pointing
    // elsewhere does not silently corrupt the transform.
    reconfig_data_format_srca(cb);
    pack_reconfig_data_format(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, 0, 0);
    t(0);
    tile_regs_commit();

    // Pop BEFORE reserve_back. The pop releases capacity in `cb`, which the
    // reserve_back then claims. Reversing the order would deadlock on a
    // 1-page CB (reserve waits for free capacity that pop hasn't released).
    cb_pop_front(cb, onetile);
    cb_reserve_back(cb, onetile);

    tile_regs_wait();
    pack_tile(0, cb);
    tile_regs_release();

    cb_push_back(cb, onetile);
}

}  // namespace compute_kernel_lib
