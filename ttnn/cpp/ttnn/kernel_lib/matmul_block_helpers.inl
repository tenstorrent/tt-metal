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
    bool transpose,
    typename PostComputeFn>
ALWI void matmul_block(
    uint32_t block_w,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t num_k_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t batch,
    PostComputeFn post_compute) {

    // Compile-time validation
    static_assert(in0_cb != out_cb, "matmul_block: in0_cb and out_cb must be different CBs");
    static_assert(in1_cb != out_cb, "matmul_block: in1_cb and out_cb must be different CBs");
    static_assert(in0_cb < 32, "matmul_block: in0_cb must be less than 32");
    static_assert(in1_cb < 32, "matmul_block: in1_cb must be less than 32");
    static_assert(out_cb < 32, "matmul_block: out_cb must be less than 32");
    static_assert(interm_cb < 32, "matmul_block: interm_cb must be less than 32");

    // Derive dependent quantities from independent parameters
    const uint32_t out_num_tiles = out_subblock_h * out_subblock_w;
    const uint32_t in0_subblock_num_tiles = out_subblock_h * block_w;
    const uint32_t in0_block_num_tiles = in0_subblock_num_tiles * in0_num_subblocks;
    const uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    const uint32_t in1_block_num_tiles = out_subblock_w * block_w * in1_num_subblocks;

    // Runtime validation
    ASSERT(block_w > 0);
    ASSERT(in0_num_subblocks > 0);
    ASSERT(in1_num_subblocks > 0);
    ASSERT(num_k_blocks > 0);
    ASSERT(out_subblock_h > 0);
    ASSERT(out_subblock_w > 0);
    ASSERT(batch > 0);

    // Verify output sub-block fits in DST registers (capacity depends on sync mode and FP32 accum)
    ASSERT(out_num_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT);

    // Verify CB capacity
    PACK(ASSERT(get_cb_num_pages(in0_cb) >= in0_block_num_tiles));
    PACK(ASSERT(get_cb_num_pages(in1_cb) >= in1_block_num_tiles));
    PACK(ASSERT(get_cb_num_pages(out_cb) >= out_num_tiles));

    // Data format reconfiguration
    reconfig_data_format_srca(in0_cb);
    reconfig_data_format_srcb(in1_cb);
    pack_reconfig_data_format(out_cb);

    // Init — use mm_block_init for hardware block-level matmul optimization.
    // Configure packer with interm_cb because during K-blocking we spill
    // partial results to the intermediate buffer.
    mm_block_init(in0_cb, in1_cb, interm_cb, transpose, out_subblock_w, out_subblock_h, block_w);

    for (uint32_t b = 0; b < batch; b++) {
        bool spill = num_k_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_num_tiles;

        for (uint32_t block = 0; block < num_k_blocks; block++) {
            bool last_out = block == (num_k_blocks - 1);

            // Wait for full input blocks
            cb_wait_front(in0_cb, in0_block_num_tiles);
            cb_wait_front(in1_cb, in1_block_num_tiles);

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    acquire_dst();

                    // Reload partial results from intermediate CB if accumulating across blocks.
                    if (enable_reload) {
                        copy_tile_to_dst_init_short_with_dt(in1_cb, interm_cb);
                        cb_wait_front(interm_cb, out_num_tiles);
                        for (uint32_t i = 0; i < out_num_tiles; i++) {
                            copy_tile(interm_cb, i, i);
                        }
                        cb_pop_front(interm_cb, out_num_tiles);
                        mm_block_init_short_with_dt(
                            in0_cb, in1_cb, interm_cb, transpose, out_subblock_w, out_subblock_h, block_w);
                    }

                    // Compute output sub-block using hardware block matmul.
                    // matmul_block LLK handles the sub-block tile iteration internally
                    // (out_subblock_h × out_subblock_w), so we only iterate over the inner K dimension.
                    uint32_t dst_index = 0;
                    uint32_t in0_index = in0_index_subblock_offset;
                    uint32_t in1_index = in1_index_subblock_offset;
                    for (uint32_t inner_dim = 0; inner_dim < block_w; inner_dim++) {
                        // Explicit ckernel:: to call the LLK matmul_block function,
                        // not this helper (which has the same name in a different namespace).
                        ckernel::matmul_block(
                            in0_cb,
                            in1_cb,
                            in0_index,
                            in1_index,
                            dst_index,
                            transpose,
                            out_subblock_w,
                            out_subblock_h,
                            block_w);
                        in0_index++;
                        in1_index += in1_per_core_w;
                    }

                    if (last_out) {
                        // Apply optional post-compute operation (e.g., SFPU activation)
                        // on DST tiles before packing.
                        post_compute(out_num_tiles);

                        // Final block: pack to output buffer
                        cb_reserve_back(out_cb, out_num_tiles);
                        for (uint32_t i = 0; i < out_num_tiles; i++) {
                            pack_tile(i, out_cb);
                        }
                        cb_push_back(out_cb, out_num_tiles);
                    } else {
                        // Reserve output space to prevent interm from overwriting
                        // (out_cb and interm_cb share memory)
                        if (block == 0) {
                            cb_reserve_back(out_cb, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_num_tiles;
                        }
                        // Spill partial result to intermediate buffer
                        cb_reserve_back(interm_cb, out_num_tiles);
                        for (uint32_t i = 0; i < out_num_tiles; i++) {
                            pack_tile(i, interm_cb);
                        }
                        cb_push_back(interm_cb, out_num_tiles);
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
