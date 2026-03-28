// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers.hpp"

/**
 * @file matmul_block_fused_bias_helpers.inl
 * @brief Implementation of matmul_block_fused_bias helper function.
 *
 * Combines mm_block_init + matmul_block LLK with a post-matmul bias addition
 * phase using add_bcast_rows. This file should only be included by
 * matmul_block_fused_bias_helpers.hpp.
 */

namespace compute_kernel_lib {

template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t interm_cb,
    uint32_t bias_cb,
    matmul_block_fused_bias_config::InitUninitMode init_uninit_mode,
    matmul_block_fused_bias_config::ReconfigureRegisterDatatypeMode reconfig_mode,
    bool transpose,
    typename PostComputeFn>
ALWI void matmul_block_fused_bias(
    matmul_block_fused_bias_config::In0BlockParams in0,
    matmul_block_fused_bias_config::In1BlockParams in1,
    uint32_t num_blocks,
    matmul_block_fused_bias_config::OutSubblockParams out,
    uint32_t batch,
    PostComputeFn post_compute) {

    // Compile-time validation
    static_assert(in0_cb != out_cb, "matmul_block_fused_bias: in0_cb and out_cb must be different CBs");
    static_assert(in1_cb != out_cb, "matmul_block_fused_bias: in1_cb and out_cb must be different CBs");
    static_assert(in0_cb < 32, "matmul_block_fused_bias: in0_cb must be less than 32");
    static_assert(in1_cb < 32, "matmul_block_fused_bias: in1_cb must be less than 32");
    static_assert(out_cb < 32, "matmul_block_fused_bias: out_cb must be less than 32");
    static_assert(interm_cb < 32, "matmul_block_fused_bias: interm_cb must be less than 32");
    static_assert(bias_cb < 32, "matmul_block_fused_bias: bias_cb must be less than 32");

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
        (reconfig_mode == matmul_block_fused_bias_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure) ||
        (reconfig_mode == matmul_block_fused_bias_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    constexpr bool use_pack_reconfig =
        (reconfig_mode == matmul_block_fused_bias_config::ReconfigureRegisterDatatypeMode::PackReconfigure) ||
        (reconfig_mode == matmul_block_fused_bias_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    if constexpr (use_unpack_reconfig) {
        reconfig_data_format_srca(in0_cb);
        reconfig_data_format_srcb(in1_cb);
    }

    if constexpr (use_pack_reconfig) {
        pack_reconfig_data_format(interm_cb);
    }

    // Init — use mm_block_init for hardware block-level matmul optimization.
    // Configure packer with interm_cb because all matmul results go to intermediate
    // buffer first, then bias-add reads from there.
    if constexpr (
        init_uninit_mode == matmul_block_fused_bias_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == matmul_block_fused_bias_config::InitUninitMode::InitOnly) {
        mm_block_init(in0_cb, in1_cb, interm_cb, transpose, out.w, out.h, in0.block_w);
    }

    for (uint32_t b = 0; b < batch; b++) {
        bool spill = num_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out.num_tiles;

        // ── Phase 1: Matmul with K-blocking ──────────────────────────────
        // Always pack results to interm_cb (both intermediate spill and final
        // matmul output), because bias-add will read from interm_cb.
        for (uint32_t block = 0; block < num_blocks; block++) {
            bool last_out = block == (num_blocks - 1);

            // Wait for full input blocks
            cb_wait_front(in0_cb, in0.block_num_tiles);
            cb_wait_front(in1_cb, in1.block_num_tiles);

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
                    uint32_t dst_index = 0;
                    uint32_t in0_index = in0_index_subblock_offset;
                    uint32_t in1_index = in1_index_subblock_offset;
                    for (uint32_t inner_dim = 0; inner_dim < in0.block_w; inner_dim++) {
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
                        // Final K-block: pack to interm_cb for bias-add phase.
                        // No SFPU activation here — it's applied after bias add.
                        cb_reserve_back(interm_cb, out.num_tiles);
                        for (uint32_t i = 0; i < out.num_tiles; i++) {
                            pack_tile(i, interm_cb);
                        }
                        cb_push_back(interm_cb, out.num_tiles);
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

            cb_pop_front(in0_cb, in0.block_num_tiles);
            cb_pop_front(in1_cb, in1.block_num_tiles);
        }

        // ── Phase 2: Bias addition with row broadcast ────────────────────
        // Reconfigure data formats from matmul to bias-add operation.
        // The four-argument reconfig_data_format sets both srcA and srcB:
        //   srcA: from in1_cb format → interm_cb format (matmul partials)
        //   srcB: from in0_cb format → bias_cb format
        reconfig_data_format(in1_cb, interm_cb, in0_cb, bias_cb);
        pack_reconfig_data_format(out_cb);
        add_bcast_rows_init_short(interm_cb, bias_cb);

        // Wait for bias tiles — one per output column tile.
        cb_wait_front(bias_cb, in1.per_core_w);

        int bias_subblock_offset = 0;
        for (uint32_t in0_subblock = 0; in0_subblock < in0.num_subblocks; in0_subblock++) {
            bias_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < in1.num_subblocks; in1_subblock++) {
                // Wait for matmul output sub-block from interm_cb.
                cb_wait_front(interm_cb, out.num_tiles);
                tile_regs_acquire();

                // Add bias with row broadcast: each row of the output sub-block
                // gets the same bias tiles (one per column tile).
                for (uint32_t j = 0, tile_idx = 0; j < out.h; j++) {
                    uint32_t bcast_tile_idx = bias_subblock_offset;
                    for (uint32_t k = 0; k < out.w; k++, tile_idx++) {
                        add_tiles_bcast_rows(interm_cb, bias_cb, tile_idx, bcast_tile_idx, tile_idx);
                        bcast_tile_idx++;
                    }
                }

                // Apply optional post-compute operation (e.g., SFPU activation)
                // on DST tiles after bias addition, before packing.
                post_compute(out.num_tiles);

                tile_regs_commit();

                // Pop interm BEFORE reserving out — critical because interm_cb and
                // out_cb share L1 memory. Data is already in DST registers, so
                // the interm CB pages can be released to make room for out_cb.
                cb_pop_front(interm_cb, out.num_tiles);

                // Pack final output to out_cb.
                cb_reserve_back(out_cb, out.num_tiles);
                tile_regs_wait();
                for (uint32_t i = 0; i < out.num_tiles; i++) {
                    pack_tile(i, out_cb);
                }
                tile_regs_release();
                cb_push_back(out_cb, out.num_tiles);

                bias_subblock_offset += out.w;
            }
        }

        // Reconfigure back to matmul format for next batch iteration.
        if (b < batch - 1) {
            reconfig_data_format(interm_cb, in1_cb, bias_cb, in0_cb);
            pack_reconfig_data_format(interm_cb);
            mm_block_init_short(in0_cb, in1_cb, transpose, out.w, out.h, in0.block_w);
        }
    }
}

}  // namespace compute_kernel_lib
