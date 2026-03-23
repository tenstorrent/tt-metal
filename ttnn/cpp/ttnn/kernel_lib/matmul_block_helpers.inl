// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers.hpp"

/**
 * @file matmul_block_helpers.inl
 * @brief Implementation of matmul_block helper function.
 *
 * Wraps the sub-blocked matmul pattern from bmm_large_block_zm.cpp.
 * This file should only be included by matmul_block_helpers.hpp.
 */

namespace compute_kernel_lib {

template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t interm_cb,
    matmul_block_config::InitUninitMode init_uninit_mode,
    matmul_block_config::ReconfigureRegisterDatatypeMode reconfig_mode>
ALWI void matmul_block(
    uint32_t in0_block_w,
    uint32_t in0_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in0_subblock_num_tiles,
    uint32_t in1_num_subblocks,
    uint32_t in1_block_num_tiles,
    uint32_t in1_per_core_w,
    uint32_t num_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_subblock_num_tiles,
    uint32_t batch) {

    // Compile-time validation
    static_assert(in0_cb != out_cb, "matmul_block: in0_cb and out_cb must be different CBs");
    static_assert(in1_cb != out_cb, "matmul_block: in1_cb and out_cb must be different CBs");
    static_assert(in0_cb < 32, "matmul_block: in0_cb must be less than 32");
    static_assert(in1_cb < 32, "matmul_block: in1_cb must be less than 32");
    static_assert(out_cb < 32, "matmul_block: out_cb must be less than 32");
    static_assert(interm_cb < 32, "matmul_block: interm_cb must be less than 32");

    // Runtime validation
    ASSERT(in0_block_w > 0);
    ASSERT(in0_num_subblocks > 0);
    ASSERT(in1_num_subblocks > 0);
    ASSERT(num_blocks > 0);
    ASSERT(out_subblock_h > 0);
    ASSERT(out_subblock_w > 0);
    ASSERT(batch > 0);

    // Data format reconfiguration (applied before init)
    constexpr bool use_unpack_reconfig =
        (reconfig_mode == matmul_block_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure) ||
        (reconfig_mode == matmul_block_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    constexpr bool use_pack_reconfig =
        (reconfig_mode == matmul_block_config::ReconfigureRegisterDatatypeMode::PackReconfigure) ||
        (reconfig_mode == matmul_block_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    if constexpr (use_unpack_reconfig) {
        reconfig_data_format_srca(in0_cb);
        reconfig_data_format_srcb(in1_cb);
    }

    if constexpr (use_pack_reconfig) {
        pack_reconfig_data_format(out_cb);
    }

    // Init
    if constexpr (
        init_uninit_mode == matmul_block_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == matmul_block_config::InitUninitMode::InitOnly) {
        mm_init(in0_cb, in1_cb, out_cb);
    }

    for (uint32_t b = 0; b < batch; b++) {
        bool spill = num_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        for (uint32_t block = 0; block < num_blocks; block++) {
            bool last_out = block == (num_blocks - 1);

            // Wait for full input blocks
            cb_wait_front(in0_cb, in0_block_num_tiles);
            cb_wait_front(in1_cb, in1_block_num_tiles);

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    acquire_dst();

                    // Reload partial results from intermediate CB if accumulating across blocks
                    if (enable_reload) {
                        copy_tile_to_dst_init_short(interm_cb);
                        cb_wait_front(interm_cb, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(interm_cb, i, i);
                        }
                        cb_pop_front(interm_cb, out_subblock_num_tiles);
                        mm_init_short(in0_cb, in1_cb);
                    }

                    // Compute output sub-block from in0_subblock x in1_subblock
                    int dst_index = 0;
                    int in0_index_h_offset = 0;
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            int in1_index_inner_dim_offset = 0;
                            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                matmul_tiles(in0_cb, in1_cb, in0_index, in1_index, dst_index);
                                in1_index_inner_dim_offset += in1_per_core_w;
                            }
                            dst_index++;
                        }
                        in0_index_h_offset += in0_block_w;
                    }

                    if (last_out) {
                        // Final block: pack to output buffer
                        cb_reserve_back(out_cb, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, out_cb);
                        }
                        cb_push_back(out_cb, out_subblock_num_tiles);
                    } else {
                        // Reserve output space to prevent interm from overwriting
                        // (out_cb and interm_cb share memory)
                        if (block == 0) {
                            cb_reserve_back(out_cb, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_subblock_num_tiles;
                        }
                        // Spill partial result to intermediate buffer
                        cb_reserve_back(interm_cb, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, interm_cb);
                        }
                        cb_push_back(interm_cb, out_subblock_num_tiles);
                    }

                    release_dst();
                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if (spill) {
                enable_reload = true;
            }

            cb_pop_front(in0_cb, in0_block_num_tiles);
            cb_pop_front(in1_cb, in1_block_num_tiles);
        }
    }
}

}  // namespace compute_kernel_lib
