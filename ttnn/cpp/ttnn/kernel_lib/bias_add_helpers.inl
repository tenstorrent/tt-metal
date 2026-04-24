// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file bias_add_helpers.inl
 * @brief Implementation of add_bias_bcast_rows helper function.
 *
 * Row-broadcast bias addition matching the production kernel's bias phase
 * (bmm_large_block_zm_fused_bias_activation.cpp).
 * This file should only be included by bias_add_helpers.hpp.
 */

namespace compute_kernel_lib {

template <
    uint32_t partials_cb,
    uint32_t bias_cb,
    uint32_t out_cb,
    BiasBroadcast broadcast,
    OutputLayout output_layout,
    typename PostBiasFn>
ALWI void add_bias_bcast_rows(BiasAddShape shape, PostBiasFn post_bias) {

    // Compile-time validation
    static_assert(partials_cb < 32, "add_bias_bcast_rows: partials_cb must be less than 32");
    static_assert(bias_cb < 32, "add_bias_bcast_rows: bias_cb must be less than 32");
    static_assert(out_cb < 32, "add_bias_bcast_rows: out_cb must be less than 32");

    // Hoist shape fields so the existing body reads unchanged.
    const uint32_t in0_num_subblocks = shape.in0_num_subblocks;
    const uint32_t in1_num_subblocks = shape.in1_num_subblocks;
    const uint32_t out_subblock_h = shape.out_subblock_h;
    const uint32_t out_subblock_w = shape.out_subblock_w;
    uint32_t out_row_width = shape.out_row_width;

    const uint32_t out_num_tiles = out_subblock_h * out_subblock_w;

    // Runtime validation
    ASSERT(in0_num_subblocks > 0);
    ASSERT(in1_num_subblocks > 0);
    ASSERT(out_subblock_h > 0);
    ASSERT(out_subblock_w > 0);
    ASSERT(out_num_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT);

    // Format reconfig + init. Caller owns bias_cb wait/pop lifecycle (reader may push
    // bias only once across multiple iterations).
    reconfig_data_format_srca(partials_cb);
    reconfig_data_format_srcb(bias_cb);
    pack_reconfig_data_format(out_cb);
    if constexpr (broadcast == BiasBroadcast::RowBroadcast) {
        add_bcast_rows_init_short(partials_cb, bias_cb);
    } else {
        add_tiles_init(partials_cb, bias_cb);
    }

    if constexpr (output_layout == OutputLayout::RowMajor) {
        // Row-major layout: upstream matmul_block pushes one M-row-group
        // (out_subblock_h × out_row_width tiles) per in0_subblock. Tile at row r, column
        // sbw*out_subblock_w + c sits at front+r*out_row_width + sbw*out_subblock_w + c.
        // Bias reads at those absolute positions, computes DST for one N-subblock at a
        // time, and packs back to the output CB at the same row-major positions.
        if (out_row_width == 0) {
            out_row_width = out_subblock_w * in1_num_subblocks;
        }
        const uint32_t row_group_tiles = out_subblock_h * out_row_width;

        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
            cb_wait_front(partials_cb, row_group_tiles);
            cb_reserve_back(out_cb, row_group_tiles);

            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                const uint32_t col_base = in1_subblock * out_subblock_w;

                tile_regs_acquire();
                {
                    uint32_t dst_idx = 0;
                    for (uint32_t r = 0; r < out_subblock_h; r++) {
                        const uint32_t partial_row_pos = r * out_row_width + col_base;
                        for (uint32_t c = 0; c < out_subblock_w; c++) {
                            if constexpr (broadcast == BiasBroadcast::RowBroadcast) {
                                add_tiles_bcast_rows(
                                    partials_cb,
                                    bias_cb,
                                    partial_row_pos + c,
                                    col_base + c,
                                    dst_idx);
                            } else {
                                add_tiles(
                                    partials_cb,
                                    bias_cb,
                                    partial_row_pos + c,
                                    col_base + c,
                                    dst_idx);
                            }
                            dst_idx++;
                        }
                    }
                }

                post_bias(out_num_tiles);
                tile_regs_commit();
                tile_regs_wait();

                // Pack DST back to out_cb at the same row-major positions. For h=1 the
                // absolute-offset form collapses to a sequential column run within the
                // row-group.
                if (out_subblock_h == 1) {
                    for (uint32_t c = 0; c < out_subblock_w; c++) {
                        pack_tile<true>(c, out_cb, col_base + c);
                    }
                } else {
                    uint32_t dst_idx = 0;
                    for (uint32_t r = 0; r < out_subblock_h; r++) {
                        const uint32_t pack_row_pos = r * out_row_width + col_base;
                        for (uint32_t c = 0; c < out_subblock_w; c++) {
                            pack_tile<true>(dst_idx, out_cb, pack_row_pos + c);
                            dst_idx++;
                        }
                    }
                }
                tile_regs_release();
            }

            cb_pop_front(partials_cb, row_group_tiles);
            cb_push_back(out_cb, row_group_tiles);
        }
    } else {
        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
            int in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                cb_wait_front(partials_cb, out_num_tiles);
                tile_regs_acquire();

                // Bias addition: row-broadcast (one bias row per column, broadcast across M)
                // or elementwise (bias has multiple M rows).
                for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
                    uint32_t bias_tile_idx = in1_index_subblock_offset;
                    for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
                        if constexpr (broadcast == BiasBroadcast::RowBroadcast) {
                            add_tiles_bcast_rows(partials_cb, bias_cb, i, bias_tile_idx, i);
                        } else {
                            add_tiles(partials_cb, bias_cb, i, bias_tile_idx, i);
                        }
                        bias_tile_idx++;
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
    }

    // NOTE: does NOT pop bias_cb. Caller manages bias tile lifetime.
}

}  // namespace compute_kernel_lib
