// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file bias_add_helpers.inl
 * @brief Implementation of add_bias_bcast_rows helper function.
 *
 * Row-broadcast bias addition matching the production kernel's bias phase
 * (bmm_large_block_zm_fused_bias_activation.cpp lines 404-462).
 * This file should only be included by bias_add_helpers.hpp.
 */

namespace compute_kernel_lib {

template <
    uint32_t partials_cb,
    uint32_t bias_cb,
    uint32_t out_cb,
    typename PostBiasFn>
ALWI void add_bias_bcast_rows(
    const uint32_t in0_num_subblocks,
    const uint32_t in1_num_subblocks,
    const uint32_t out_subblock_h,
    const uint32_t out_subblock_w,
    const uint32_t bias_width_tiles,
    PostBiasFn post_bias) {

    // Compile-time validation
    static_assert(partials_cb < 32, "add_bias_bcast_rows: partials_cb must be less than 32");
    static_assert(bias_cb < 32, "add_bias_bcast_rows: bias_cb must be less than 32");
    static_assert(out_cb < 32, "add_bias_bcast_rows: out_cb must be less than 32");

    const uint32_t out_num_tiles = out_subblock_h * out_subblock_w;

    // Runtime validation
    ASSERT(in0_num_subblocks > 0);
    ASSERT(in1_num_subblocks > 0);
    ASSERT(out_subblock_h > 0);
    ASSERT(out_subblock_w > 0);
    ASSERT(bias_width_tiles > 0);
    ASSERT(out_num_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT);

    // Format reconfig for bias phase
    reconfig_data_format_srca(partials_cb);
    reconfig_data_format_srcb(bias_cb);
    pack_reconfig_data_format(out_cb);

    // Init bias broadcast
    add_bcast_rows_init_short(partials_cb, bias_cb);

    // Wait for bias tiles (no-op if already available)
    cb_wait_front(bias_cb, bias_width_tiles);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
        int in1_index_subblock_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
            cb_wait_front(partials_cb, out_num_tiles);
            tile_regs_acquire();

            // Row-broadcast bias addition
            for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
                uint32_t bcast_tile_idx = in1_index_subblock_offset;
                for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
                    add_tiles_bcast_rows(partials_cb, bias_cb, i, bcast_tile_idx, i);
                    bcast_tile_idx++;
                }
            }

            // PostBiasFn fires BEFORE commit (matches production kernel's SFPU placement)
            post_bias(out_num_tiles);

            tile_regs_commit();
            cb_pop_front(partials_cb, out_num_tiles);

            // Pack out to output buffer
            cb_reserve_back(out_cb, out_num_tiles);
            tile_regs_wait();
            for (uint32_t i = 0; i < out_num_tiles; i++) {
                pack_tile(i, out_cb);
            }
            tile_regs_release();
            cb_push_back(out_cb, out_num_tiles);

            in1_index_subblock_offset += out_subblock_w;
        }
    }

    // NOTE: does NOT pop bias_cb. Caller manages bias tile lifetime.
}

}  // namespace compute_kernel_lib
