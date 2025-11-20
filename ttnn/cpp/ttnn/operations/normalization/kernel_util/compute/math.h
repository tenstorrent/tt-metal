// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file math.h
 * @brief Generic math utilities for compute kernels.
 */

#pragma once

#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/reduce.h"
#include "dprint_pages.h"

namespace norm::kernel_util::compute::math {

/**
 * @brief Compute the row-wise mean of an entire input CB.
 * @param cb_in The input CB to compute mean of
 * @param cb_scalar The CB containing a scalar tile
 * to scale the sum by
 * @param cb_interm The intermediate CB to pack the sum
 * to before reducing it
 * @param cb_out The output CB to store the mean in
 * before processing the next block. Must be an even number
 * @param num_tiles The number of tiles in the mean
 * @tparam pop_input Whether to pop the input CB after processing
 * @tparam FP32_REDUCE Whether to reduce the sum in FP32 precision
 *
 * @note dst0 is used to accumuate the sum, so it
 * will be overwritten here
 */
template <bool pop_input, bool FP32_REDUCE>
inline void row_wise_mean(uint32_t cb_in, uint32_t cb_scalar, uint32_t cb_interm, uint32_t cb_out, uint32_t num_tiles) {
    constexpr uint32_t dst0 = 0;
    const uint32_t num_pairs = num_tiles / 2;
    const uint32_t has_remainder_tile = num_tiles % 2 != 0;

    reconfig_data_format(cb_in, cb_in);
    fill_tile_init();
    add_tiles_init(cb_in, cb_in, true);

    tile_regs_acquire();

    // Zero out dst0
    fill_tile(dst0, 0.0f);

    // Accumulate pair-wise sums of the input CB into dst0
    for (uint32_t i = 0; i < num_pairs; i++) {
        if constexpr (pop_input) {
            cb_wait_front(cb_in, 2);
            add_tiles(cb_in, cb_in, 0, 1, dst0);
            cb_pop_front(cb_in, 2);
        } else {
            cb_wait_front(cb_in, i * 2 + 2);
            add_tiles(cb_in, cb_in, i * 2, i * 2 + 1, dst0);
        }
    }

    if (has_remainder_tile) {
        // Add the final tile to the sum
        if constexpr (pop_input) {
            cb_wait_front(cb_in, 1);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_in);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_in, 0, dst0);
            cb_pop_front(cb_in, 1);
        } else {
            cb_wait_front(cb_in, num_tiles);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_in);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_in, num_tiles - 1, dst0);
        }
    }

    tile_regs_commit();
    tile_regs_wait();

    // Pack the sum into the intermediate CB
    cb_reserve_back(cb_interm, 1);
    pack_reconfig_data_format(cb_interm);
    pack_tile(dst0, cb_interm);

    tile_regs_release();

    cb_push_back(cb_interm, 1);

    // Reduce the intermediate CB tile and pack
    // to output CB
    cb_wait_front(cb_interm, 1);
    cb_wait_front(cb_scalar, 1);
    reconfig_data_format(cb_interm, cb_scalar);
    reduce_init<REDUCE_OP, REDUCE_DIM, FP32_REDUCE>(cb_interm, cb_scalar, cb_out);
    tile_regs_acquire();
    reduce_tile<REDUCE_OP, REDUCE_DIM, FP32_REDUCE>(cb_interm, cb_scalar, 0, 0, dst0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_reconfig_data_format(cb_out);
    pack_tile(dst0, cb_out);
    reduce_uninit<FP32_REDUCE>();
    tile_regs_release();

    cb_pop_front(cb_interm, 1);
    cb_push_back(cb_out, 1);
}
}  // namespace norm::kernel_util::compute::math
