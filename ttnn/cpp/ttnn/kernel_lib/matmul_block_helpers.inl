// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers_compute.hpp"

/**
 * @file matmul_block_helpers.inl
 * @brief Implementation of matmul_block helper function.
 *
 * Uses mm_block_init + matmul_block LLK for hardware-optimized block matmul.
 * Caller must call mm_block_init before invoking this helper.
 * This file should only be included by matmul_block_helpers.hpp.
 */

namespace compute_kernel_lib {

template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t interm_cb,
    bool transpose,
    bool packer_l1_acc,
    bool pack_last_to_interm,
    typename PostComputeFn,
    typename PreKBlockFn>
ALWI void matmul_block(
    const uint32_t block_w,
    const uint32_t in0_num_subblocks,
    const uint32_t in1_num_subblocks,
    const uint32_t num_k_blocks,
    const uint32_t out_subblock_h,
    const uint32_t out_subblock_w,
    const uint32_t batch,
    PostComputeFn post_compute,
    PreKBlockFn pre_k_block) {

    // Compile-time validation
    static_assert(in0_cb != out_cb, "matmul_block: in0_cb and out_cb must be different CBs");
    static_assert(in1_cb != out_cb, "matmul_block: in1_cb and out_cb must be different CBs");
    static_assert(in0_cb < 32, "matmul_block: in0_cb must be less than 32");
    static_assert(in1_cb < 32, "matmul_block: in1_cb must be less than 32");
    static_assert(out_cb < 32, "matmul_block: out_cb must be less than 32");
    static_assert(interm_cb < 32, "matmul_block: interm_cb must be less than 32");

    constexpr bool hw_relu = std::is_same_v<PostComputeFn, matmul_block_config::HwRelu>;
    static_assert(
        !(hw_relu && pack_last_to_interm),
        "matmul_block: HwRelu cannot be used with pack_last_to_interm (relu should be applied after bias)");

    // Derive dependent quantities from independent parameters
    const uint32_t out_num_tiles = out_subblock_h * out_subblock_w;
    const uint32_t in0_subblock_num_tiles = out_subblock_h * block_w;
    const uint32_t in0_block_num_tiles = in0_subblock_num_tiles * in0_num_subblocks;
    const uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    const uint32_t in1_block_num_tiles = out_subblock_w * block_w * in1_num_subblocks;
    const uint32_t out_block_num_tiles = out_num_tiles * in0_num_subblocks * in1_num_subblocks;

    // Runtime validation
    ASSERT(block_w > 0);
    ASSERT(in0_num_subblocks > 0);
    ASSERT(in1_num_subblocks > 0);
    ASSERT(num_k_blocks > 0);
    ASSERT(out_subblock_h > 0);
    ASSERT(out_subblock_w > 0);
    ASSERT(batch > 0);

    // Verify output sub-block fits in DST registers
    ASSERT(out_num_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT);

    // Verify CB capacity
    PACK(ASSERT(get_cb_num_pages(in0_cb) >= in0_block_num_tiles));
    PACK(ASSERT(get_cb_num_pages(in1_cb) >= in1_block_num_tiles));
    PACK(ASSERT(get_cb_num_pages(out_cb) >= out_num_tiles));

    // Compile-time pack target: last K-block goes to interm or out
    constexpr uint32_t pack_target = pack_last_to_interm ? interm_cb : out_cb;

    // Initialize matmul operation (caller must have called compute_kernel_hw_startup or mm_block_init)
    mm_block_init_short(in0_cb, in1_cb, transpose, out_subblock_w, out_subblock_h, block_w);

    for (uint32_t b = 0; b < batch; b++) {
        bool spill = num_k_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_num_tiles;

        for (uint32_t block = 0; block < num_k_blocks; block++) {
            bool last_out = block == (num_k_blocks - 1);

            // HwRelu: configure packer RELU on last block when packing directly to output
            if constexpr (hw_relu) {
                if (last_out) {
                    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
                }
            }

            // PreKBlockFn: per-K-block preprocessing (e.g., in0_transpose)
            pre_k_block(block, num_k_blocks, last_out);

            // Wait for full input blocks
            cb_wait_front(in0_cb, in0_block_num_tiles);
            cb_wait_front(in1_cb, in1_block_num_tiles);

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    tile_regs_acquire();

                    // Reload partial results from intermediate CB if accumulating across blocks.
                    if (enable_reload) {
                        copy_tile_to_dst_init_short_with_dt(in1_cb, interm_cb);
                        cb_wait_front(interm_cb, out_num_tiles);
                        copy_block_matmul_partials(interm_cb, 0, 0, out_num_tiles);
                        cb_pop_front(interm_cb, out_num_tiles);
                        mm_block_init_short_with_dt(
                            in0_cb, in1_cb, interm_cb, transpose, out_subblock_w, out_subblock_h, block_w);
                    }

                    // Compute output sub-block using hardware block matmul.
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
                        // Apply optional SFPU activation (skipped for HwRelu — packer handles it)
                        if constexpr (!hw_relu) {
                            post_compute(out_num_tiles);
                        }

                        tile_regs_commit();
                        cb_reserve_back(pack_target, out_num_tiles);
                        tile_regs_wait();

                        // Pack format reconfig (handles FP32_DEST_ACC_EN automatically)
                        if constexpr (packer_l1_acc || get_fp32_dest_acc_enabled()) {
                            PACK((pack_reconfig_data_format(pack_target)));
                        }

                        // L1_ACC toggle on last block
                        if constexpr (packer_l1_acc) {
                            if constexpr (pack_last_to_interm) {
                                // FUSE_BIAS path: L1 accumulates across all blocks.
                                // Block 0 = no accumulation, block 1+ = accumulate.
                                if (block == 0) {
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                } else {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
                            } else {
                                // No bias: disable L1_ACC for final output
                                PACK((llk_pack_reconfig_l1_acc(0)));
                            }
                        }

                        pack_tile_block(0, pack_target, out_num_tiles);
                        tile_regs_release();
                        cb_push_back(pack_target, out_num_tiles);

                    } else {
                        // Not last block: spill partial result to intermediate buffer
                        tile_regs_commit();

                        // Reserve output space to prevent interm from overwriting
                        // (out_cb and interm_cb share memory)
                        if (block == 0) {
                            cb_reserve_back(out_cb, out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_num_tiles;
                        }

                        cb_reserve_back(interm_cb, out_num_tiles);
                        tile_regs_wait();

                        if constexpr (packer_l1_acc) {
                            if (block == 0) {
                                PACK((llk_pack_reconfig_l1_acc(0)));
                            } else {
                                PACK((llk_pack_reconfig_l1_acc(1)));
                            }
                        }

                        pack_tile_block(0, interm_cb, out_num_tiles);
                        tile_regs_release();
                        cb_push_back(interm_cb, out_num_tiles);
                    }

                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            // Post-subblock-loop: manage reload state
            if constexpr (packer_l1_acc) {
                if constexpr (pack_last_to_interm) {
                    // L1_ACC + bias: advance FIFO, never reload
                    if (block < num_k_blocks - 1) {
                        cb_wait_front(interm_cb, out_block_num_tiles);
                        cb_pop_front(interm_cb, out_block_num_tiles);
                    }
                    enable_reload = false;
                } else {
                    // L1_ACC + no bias: advance FIFO, reload on K-1
                    if (num_k_blocks >= 2 && block < num_k_blocks - 2) {
                        cb_wait_front(interm_cb, out_block_num_tiles);
                        cb_pop_front(interm_cb, out_block_num_tiles);
                    }
                    if (block == num_k_blocks - 2) {
                        enable_reload = true;
                    }
                }
            } else {
                if (spill) {
                    enable_reload = true;
                }
            }

            cb_pop_front(in0_cb, in0_block_num_tiles);
            cb_pop_front(in1_cb, in1_block_num_tiles);
        }
    }
}

}  // namespace compute_kernel_lib
