// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers.hpp"

/**
 * @file matmul_block_helpers.inl
 * @brief Implementation of matmul_block helper function.
 *
 * Uses mm_block_init + matmul_block LLK for hardware-optimized block matmul.
 * This file should only be included by matmul_block_helpers.hpp.
 */

namespace compute_kernel_lib {

template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t interm_cb,
    matmul_block_config::InitUninitMode init_uninit_mode,
    matmul_block_config::ReconfigureRegisterDatatypeMode reconfig_mode,
    matmul_block_config::WaitPopMode wait_pop_mode,
    bool transpose,
    typename PostComputeFn>
ALWI void matmul_block(
    matmul_block_config::In0BlockParams in0,
    matmul_block_config::In1BlockParams in1,
    uint32_t num_blocks,
    matmul_block_config::OutSubblockParams out,
    uint32_t batch,
    PostComputeFn post_compute) {

    // Compile-time validation
    static_assert(in0_cb != out_cb, "matmul_block: in0_cb and out_cb must be different CBs");
    static_assert(in1_cb != out_cb, "matmul_block: in1_cb and out_cb must be different CBs");
    static_assert(in0_cb < 32, "matmul_block: in0_cb must be less than 32");
    static_assert(in1_cb < 32, "matmul_block: in1_cb must be less than 32");
    static_assert(out_cb < 32, "matmul_block: out_cb must be less than 32");
    static_assert(interm_cb < 32, "matmul_block: interm_cb must be less than 32");

    // Runtime validation
    ASSERT(in0.block_w > 0);
    ASSERT(in0.num_subblocks > 0);
    ASSERT(in1.num_subblocks > 0);
    ASSERT(num_blocks > 0);
    ASSERT(out.h > 0);
    ASSERT(out.w > 0);
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

    // Init — use mm_block_init for hardware block-level matmul optimization.
    // Configure packer with interm_cb because during K-blocking we spill
    // partial results to the intermediate buffer.
    if constexpr (
        init_uninit_mode == matmul_block_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == matmul_block_config::InitUninitMode::InitOnly) {
        mm_block_init(in0_cb, in1_cb, interm_cb, transpose, out.w, out.h, in0.block_w);
    }

    for (uint32_t b = 0; b < batch; b++) {
        bool spill = num_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out.num_tiles;

        for (uint32_t block = 0; block < num_blocks; block++) {
            bool last_out = block == (num_blocks - 1);

            // Wait for full input blocks
            if constexpr (wait_pop_mode == matmul_block_config::WaitPopMode::WaitAndPop) {
                cb_wait_front(in0_cb, in0.block_num_tiles);
                cb_wait_front(in1_cb, in1.block_num_tiles);
            }

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0.num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1.num_subblocks; in1_subblock++) {
                    acquire_dst();

                    // Reload partial results from intermediate CB if accumulating across blocks.
                    if (enable_reload) {
                        copy_tile_to_dst_init_short_with_dt(in1_cb, interm_cb);
                        cb_wait_front(interm_cb, out.num_tiles);
                        for (uint32_t i = 0; i < out.num_tiles; i++) {
                            copy_tile(interm_cb, i, i);
                        }
                        cb_pop_front(interm_cb, out.num_tiles);
                        mm_block_init_short_with_dt(
                            in0_cb, in1_cb, interm_cb, transpose, out.w, out.h, in0.block_w);
                    }

                    // Compute output sub-block using hardware block matmul.
                    // matmul_block LLK handles the sub-block tile iteration internally
                    // (out.h × out.w), so we only iterate over the inner K dimension.
                    uint32_t dst_index = 0;
                    uint32_t in0_index = in0_index_subblock_offset;
                    uint32_t in1_index = in1_index_subblock_offset;
                    for (uint32_t inner_dim = 0; inner_dim < in0.block_w; inner_dim++) {
                        // Explicit ckernel:: to call the LLK matmul_block function,
                        // not this helper (which has the same name in a different namespace).
                        ckernel::matmul_block(
                            in0_cb,
                            in1_cb,
                            in0_index,
                            in1_index,
                            dst_index,
                            transpose,
                            out.w,
                            out.h,
                            in0.block_w);
                        in0_index++;
                        in1_index += in1.per_core_w;
                    }

                    if (last_out) {
                        // Apply optional post-compute operation (e.g., SFPU activation)
                        // on DST tiles before packing.
                        post_compute(out.num_tiles);

                        // Final block: pack to output buffer
                        cb_reserve_back(out_cb, out.num_tiles);
                        for (uint32_t i = 0; i < out.num_tiles; i++) {
                            pack_tile(i, out_cb);
                        }
                        cb_push_back(out_cb, out.num_tiles);
                    } else {
                        // Reserve output space to prevent interm from overwriting
                        // (out_cb and interm_cb share memory)
                        if (block == 0) {
                            cb_reserve_back(out_cb, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out.num_tiles;
                        }
                        // Spill partial result to intermediate buffer
                        cb_reserve_back(interm_cb, out.num_tiles);
                        for (uint32_t i = 0; i < out.num_tiles; i++) {
                            pack_tile(i, interm_cb);
                        }
                        cb_push_back(interm_cb, out.num_tiles);
                    }

                    release_dst();
                    in1_index_subblock_offset += out.w;
                }
                in0_index_subblock_offset += in0.subblock_num_tiles;
            }

            if (spill) {
                enable_reload = true;
            }

            if constexpr (wait_pop_mode == matmul_block_config::WaitPopMode::WaitAndPop) {
                cb_pop_front(in0_cb, in0.block_num_tiles);
                cb_pop_front(in1_cb, in1.block_num_tiles);
            }
        }
    }
}

}  // namespace compute_kernel_lib
