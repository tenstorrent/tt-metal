// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

// =============================================================================
// detail:: building blocks
// =============================================================================

namespace detail {

template <MatmulMode mode>
ALWI void matmul_single(const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t dst_idx) {
    if constexpr (mode == MatmulMode::BLOCK) {
        ckernel::matmul_block(
            cfg.in0_cb_id, cfg.in1_cb_id,
            in0_idx, in1_idx, dst_idx,
            cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
    } else {
        ckernel::matmul_tiles(cfg.in0_cb_id, cfg.in1_cb_id, in0_idx, in1_idx, dst_idx);
    }
}

}  // namespace detail

// =============================================================================
// Initialization
// =============================================================================

template <MatmulMode mode>
ALWI void matmul_init_short(const MatmulConfig& cfg) {
    if constexpr (mode == MatmulMode::BLOCK) {
        ckernel::mm_block_init_short(
            cfg.in0_cb_id, cfg.in1_cb_id, cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
    } else {
        ckernel::mm_init_short(cfg.in0_cb_id, cfg.in1_cb_id, cfg.transpose);
    }
}

// =============================================================================
// Unified matmul helper
// =============================================================================

template <
    MatmulMode mode,
    bool packer_l1_acc,
    bool pack_last_to_interm,
    bool pack_relu,
    typename PostComputeFn,
    typename PreKBlockFn>
ALWI void matmul_blocks_absolute(
    const MatmulConfig& cfg,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t num_k_blocks,
    PostComputeFn post_compute,
    PreKBlockFn pre_k_block,
    bool retain_in0) {

    // The inner loop uses block-level indexing (matmul_block LLK).
    // TILE mode would require nested rt×ct loops with different stride logic.
    static_assert(mode == MatmulMode::BLOCK, "matmul_blocks_absolute only supports BLOCK mode");

    const uint32_t out_subblock_h = cfg.rt_dim;
    const uint32_t out_subblock_w = cfg.ct_dim;
    const uint32_t block_w = cfg.kt_dim;

    const uint32_t out_num_tiles = out_subblock_h * out_subblock_w;
    const uint32_t in0_subblock_num_tiles = out_subblock_h * block_w;
    const uint32_t in0_block_num_tiles = in0_subblock_num_tiles * in0_num_subblocks;
    const uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    const uint32_t in1_block_num_tiles = out_subblock_w * block_w * in1_num_subblocks;
    const uint32_t out_block_num_tiles = out_num_tiles * in0_num_subblocks * in1_num_subblocks;
    const uint32_t row_group_tiles = out_subblock_h * in1_per_core_w;

    // Pack target: last K-block goes to partials (for bias) or out
    const uint32_t pack_target = pack_last_to_interm ? cfg.partials_cb_id : cfg.out_cb_id;

    ASSERT(out_num_tiles <= DEST_AUTO_LIMIT);

    bool enable_reload = false;

    for (uint32_t block = 0; block < num_k_blocks; block++) {
        bool last_out = block == (num_k_blocks - 1);

        // PACK_RELU: enable on last block when packing directly to output
        if constexpr (pack_relu && !pack_last_to_interm) {
            if (last_out) {
                PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
            }
        }

        pre_k_block(block, num_k_blocks, last_out);

        // Wait for in1 block (full block per K-iteration)
        cb_wait_front(cfg.in1_cb_id, in1_block_num_tiles);

        // Progressive in0 wait
        uint32_t in0_wait_tiles = in0_subblock_num_tiles;

        int in0_index_subblock_offset = 0;
        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
            cb_wait_front(cfg.in0_cb_id, in0_wait_tiles);
            in0_wait_tiles += in0_subblock_num_tiles;

            // Reserve per-row-group for absolute-offset packing on last K-block.
            // Smaller than full-block reserve — doesn't starve shared partials_cb.
            if (last_out) {
                cb_reserve_back(pack_target, row_group_tiles);
            }

            int in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                tile_regs_acquire();

                if (enable_reload) {
                    copy_tile_to_dst_init_short_with_dt(cfg.in1_cb_id, cfg.partials_cb_id);
                    cb_wait_front(cfg.partials_cb_id, out_num_tiles);
                    copy_block_matmul_partials(cfg.partials_cb_id, 0, 0, out_num_tiles);
                    cb_pop_front(cfg.partials_cb_id, out_num_tiles);
                    ckernel::mm_block_init_short_with_dt(
                        cfg.in0_cb_id, cfg.in1_cb_id, cfg.partials_cb_id,
                        cfg.transpose, out_subblock_w, out_subblock_h, block_w);
                }

                // Compute output sub-block
                uint32_t dst_index = 0;
                uint32_t in0_index = in0_index_subblock_offset;
                uint32_t in1_index = in1_index_subblock_offset;
                for (uint32_t inner_dim = 0; inner_dim < block_w; inner_dim++) {
                    ckernel::matmul_block(
                        cfg.in0_cb_id, cfg.in1_cb_id,
                        in0_index, in1_index, dst_index,
                        cfg.transpose, out_subblock_w, out_subblock_h, block_w);
                    in0_index++;
                    in1_index += in1_per_core_w;
                }

                if (last_out) {
                    post_compute(out_num_tiles);

                    tile_regs_commit();
                    tile_regs_wait();

                    if constexpr (packer_l1_acc || get_fp32_dest_acc_enabled()) {
                        PACK((pack_reconfig_data_format(pack_target)));
                    }

                    if constexpr (packer_l1_acc) {
                        if constexpr (pack_last_to_interm) {
                            PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
                        } else {
                            PACK((llk_pack_reconfig_l1_acc(0)));
                        }
                    }

                    // Absolute-offset pack: row-major positions within row-group
                    uint32_t dst_idx = 0;
                    uint32_t col_base = in1_subblock * out_subblock_w;
                    for (uint32_t r = 0; r < out_subblock_h; r++) {
                        uint32_t row_pos = r * in1_per_core_w;
                        for (uint32_t c = 0; c < out_subblock_w; c++) {
                            pack_tile<true>(dst_idx, pack_target, row_pos + col_base + c);
                            dst_idx++;
                        }
                    }
                    tile_regs_release();

                } else {
                    // Spill to intermediate (sequential pack)
                    tile_regs_commit();
                    cb_reserve_back(cfg.partials_cb_id, out_num_tiles);
                    tile_regs_wait();

                    if constexpr (packer_l1_acc) {
                        PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
                    }

                    pack_tile_block(0, cfg.partials_cb_id, out_num_tiles);
                    tile_regs_release();
                    cb_push_back(cfg.partials_cb_id, out_num_tiles);
                }

                in1_index_subblock_offset += out_subblock_w;
            }

            // Push one row-group after all N-subblocks
            if (last_out) {
                cb_push_back(pack_target, row_group_tiles);
            }

            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

        // Post-K-block: manage reload state
        if constexpr (packer_l1_acc) {
            if constexpr (pack_last_to_interm) {
                if (block < num_k_blocks - 1) {
                    cb_wait_front(cfg.partials_cb_id, out_block_num_tiles);
                    cb_pop_front(cfg.partials_cb_id, out_block_num_tiles);
                }
                enable_reload = false;
            } else {
                if (num_k_blocks >= 2 && block < num_k_blocks - 2) {
                    cb_wait_front(cfg.partials_cb_id, out_block_num_tiles);
                    cb_pop_front(cfg.partials_cb_id, out_block_num_tiles);
                }
                if (block == num_k_blocks - 2) {
                    enable_reload = true;
                }
            }
        } else {
            if (num_k_blocks > 1) {
                enable_reload = true;
            }
        }

        // retain_in0: skip in0 pop on last K-block so caller retains the data
        // (e.g. SDPA reuses Q across K chunks). Intermediate K-blocks always pop.
        if (!retain_in0 || !last_out) {
            cb_pop_front(cfg.in0_cb_id, in0_block_num_tiles);
        }
        cb_pop_front(cfg.in1_cb_id, in1_block_num_tiles);
    }
}

// =============================================================================
// Reduce helper
// =============================================================================

template <MatmulMode mode>
ALWI void matmul_reduce_subblock_inplace(
    const MatmulConfig& cfg, uint32_t num_subblocks, uint32_t subblock_tiles, uint32_t total_in0_tiles) {
    matmul_init_short<mode>(cfg);
    reconfig_data_format(cfg.in1_cb_id, cfg.in0_cb_id);
    cb_wait_front(cfg.in1_cb_id, 1);
    cb_wait_front(cfg.out_cb_id, total_in0_tiles);

    for (uint32_t sub = 0; sub < num_subblocks; ++sub) {
        tile_regs_acquire();
        detail::matmul_single<mode>(cfg, 0, 0, 0);
        tile_regs_commit();
        cb_pop_front(cfg.out_cb_id, subblock_tiles);
        tile_regs_wait();
        for (uint32_t i = 0; i < subblock_tiles; i++) {
            pack_tile(i, cfg.out_cb_id);
        }
        tile_regs_release();
        cb_push_back(cfg.out_cb_id, subblock_tiles);
    }
}

}  // namespace compute_kernel_lib
