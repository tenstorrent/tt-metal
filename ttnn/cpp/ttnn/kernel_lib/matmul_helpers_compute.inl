// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for matmul_helpers_compute.hpp
// Do not include directly — include matmul_helpers_compute.hpp instead

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

#ifdef ARCH_BLACKHOLE
#include "api/compute/experimental/matmul_custom.h"
#endif

namespace compute_kernel_lib {

// =============================================================================
// Internal: single matmul dispatch
// =============================================================================

namespace detail {

template <MatmulMode mode>
ALWI void matmul_single(const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t dst_idx) {
    if constexpr (mode == MatmulMode::BLOCK) {
        matmul_block(
            cfg.in0_cb_id, cfg.in1_cb_id, in0_idx, in1_idx, dst_idx,
            cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
    } else {
        matmul_tiles(cfg.in0_cb_id, cfg.in1_cb_id, in0_idx, in1_idx, dst_idx);
    }
}

}  // namespace detail

// =============================================================================
// Low-Level: Initialization
// =============================================================================

template <MatmulMode mode>
ALWI void matmul_init(const MatmulConfig& cfg) {
    if constexpr (mode == MatmulMode::BLOCK) {
        mm_block_init(
            cfg.in0_cb_id, cfg.in1_cb_id, cfg.out_cb_id,
            cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
    } else {
        mm_init(cfg.in0_cb_id, cfg.in1_cb_id, cfg.out_cb_id, cfg.transpose);
    }
}

template <MatmulMode mode>
ALWI void matmul_init_short(const MatmulConfig& cfg) {
    if constexpr (mode == MatmulMode::BLOCK) {
        mm_block_init_short(
            cfg.in0_cb_id, cfg.in1_cb_id,
            cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
    } else {
        mm_init_short(cfg.in0_cb_id, cfg.in1_cb_id, cfg.transpose);
    }
}

template <MatmulMode mode>
ALWI void matmul_init_short_with_dt(const MatmulConfig& cfg, uint32_t old_in1_cb_id) {
    if constexpr (mode == MatmulMode::BLOCK) {
        mm_block_init_short_with_dt(
            cfg.in0_cb_id, cfg.in1_cb_id, old_in1_cb_id,
            cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
    } else {
        mm_init_short_with_dt(cfg.in0_cb_id, cfg.in1_cb_id, old_in1_cb_id, cfg.transpose);
    }
}

ALWI void matmul_init_short_with_both_dt(
    const MatmulConfig& cfg, uint32_t old_in0_cb_id, uint32_t old_in1_cb_id) {
    mm_block_init_short_with_both_dt(
        cfg.in0_cb_id, cfg.in1_cb_id, old_in0_cb_id, old_in1_cb_id,
        cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
}

// =============================================================================
// Low-Level: Single Tile Execution
// =============================================================================

template <MatmulMode mode>
ALWI void matmul_tile(const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t dst_idx) {
    detail::matmul_single<mode>(cfg, in0_idx, in1_idx, dst_idx);
}

#ifdef ARCH_BLACKHOLE
ALWI void matmul_tile_no_mop(const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t dst_idx) {
    matmul_block_no_mop(
        cfg.in0_cb_id, cfg.in1_cb_id, in0_idx, in1_idx, dst_idx,
        cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
}
#endif

// =============================================================================
// Mid-Level: Accumulation Patterns
// =============================================================================

template <MatmulMode mode>
ALWI void matmul_accumulate(
    const MatmulConfig& cfg,
    uint32_t in0_start,
    uint32_t in1_start,
    uint32_t dst_start,
    uint32_t count,
    uint32_t in0_stride,
    uint32_t in1_stride,
    uint32_t dst_stride) {
    for (uint32_t k = 0; k < count; ++k) {
        detail::matmul_single<mode>(cfg, in0_start + k * in0_stride, in1_start + k * in1_stride, dst_start + k * dst_stride);
    }
}

template <MatmulMode mode>
ALWI void matmul_accumulate_subblock(
    const MatmulConfig& cfg,
    uint32_t in0_offset,
    uint32_t in1_offset,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t inner_dim,
    uint32_t in1_stride) {
    uint32_t dst_index = 0;
    for (uint32_t h = 0; h < out_h; ++h) {
        for (uint32_t w = 0; w < out_w; ++w) {
            matmul_accumulate<mode>(
                cfg,
                in0_offset + h * inner_dim,
                in1_offset + w,
                dst_index,
                inner_dim,
                1,
                in1_stride,
                0);
            ++dst_index;
        }
    }
}

#ifdef ARCH_BLACKHOLE
ALWI void matmul_accumulate_no_mop(
    const MatmulConfig& cfg,
    uint32_t in0_start,
    uint32_t in1_start,
    uint32_t dst_start,
    uint32_t count,
    uint32_t in0_stride,
    uint32_t in1_stride,
    uint32_t dst_stride) {
    for (uint32_t k = 0; k < count; ++k) {
        matmul_block_no_mop(
            cfg.in0_cb_id, cfg.in1_cb_id,
            in0_start + k * in0_stride, in1_start + k * in1_stride, dst_start + k * dst_stride,
            cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
    }
}
#endif

// =============================================================================
// Mid-Level: DST Register and Pack Management
// =============================================================================

ALWI void matmul_acquire_dst() { tile_regs_acquire(); }

ALWI void matmul_pack_output(uint32_t dest_cb_id, uint32_t num_tiles) {
    tile_regs_commit();
    cb_reserve_back(dest_cb_id, num_tiles);
    tile_regs_wait();
    for (uint32_t i = 0; i < num_tiles; i++) {
        pack_tile(i, dest_cb_id);
    }
    tile_regs_release();
    cb_push_back(dest_cb_id, num_tiles);
}

ALWI void matmul_pack_partials(const MatmulConfig& cfg, uint32_t num_tiles) {
    tile_regs_commit();
    cb_reserve_back(cfg.partials_cb_id, num_tiles);
    tile_regs_wait();
    for (uint32_t i = 0; i < num_tiles; i++) {
        pack_tile(i, cfg.partials_cb_id);
    }
    tile_regs_release();
    cb_push_back(cfg.partials_cb_id, num_tiles);
}

template <MatmulMode mode>
ALWI void matmul_reload_partials(const MatmulConfig& cfg, uint32_t num_tiles) {
    copy_tile_to_dst_init_short_with_dt(cfg.in1_cb_id, cfg.partials_cb_id);
    cb_wait_front(cfg.partials_cb_id, num_tiles);
    copy_block_matmul_partials(cfg.partials_cb_id, 0, 0, num_tiles);
    cb_pop_front(cfg.partials_cb_id, num_tiles);
    // Reconfigure back to matmul mode after copy_tile changes SRCA format
    matmul_init_short_with_dt<mode>(cfg, cfg.partials_cb_id);
}

template <MatmulMode mode>
ALWI void matmul_accumulate_and_pack(
    const MatmulConfig& cfg,
    uint32_t in0_start,
    uint32_t in1_start,
    uint32_t inner_dim,
    uint32_t in1_stride,
    uint32_t dest_cb_id,
    uint32_t num_tiles,
    bool reload) {
    matmul_acquire_dst();
    if (reload) {
        matmul_reload_partials<mode>(cfg, num_tiles);
    }
    matmul_accumulate<mode>(cfg, in0_start, in1_start, 0, inner_dim, 1, in1_stride, 0);
    matmul_pack_output(dest_cb_id, num_tiles);
}

// =============================================================================
// High-Level: Full Operations
// =============================================================================

ALWI void matmul_compute_tile(const MatmulConfig& cfg, uint32_t inner_dim) {
    tile_regs_acquire();
    for (uint32_t k = 0; k < inner_dim; ++k) {
        cb_wait_front(cfg.in0_cb_id, 1);
        cb_wait_front(cfg.in1_cb_id, 1);
        detail::matmul_single<MatmulMode::TILE>(cfg, 0, 0, 0);
        cb_pop_front(cfg.in0_cb_id, 1);
        cb_pop_front(cfg.in1_cb_id, 1);
    }
    tile_regs_commit();
    cb_reserve_back(cfg.out_cb_id, 1);
    tile_regs_wait();
    pack_tile(0, cfg.out_cb_id);
    tile_regs_release();
    cb_push_back(cfg.out_cb_id, 1);
}

ALWI void matmul_inner_block(
    const MatmulConfig& cfg,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in1_block_num_tiles,
    uint32_t in1_block_w,
    bool enable_reload,
    bool last_out) {
    uint32_t out_subblock_num_tiles = cfg.ct_dim * cfg.rt_dim;
    uint32_t in0_subblock_num_tiles = cfg.rt_dim * cfg.kt_dim;

    cb_wait_front(cfg.in0_cb_id, in0_block_num_tiles);
    cb_wait_front(cfg.in1_cb_id, in1_block_num_tiles);

    uint32_t in0_index_subblock_offset = 0;
    for (uint32_t in0_sub = 0; in0_sub < in0_num_subblocks; in0_sub++) {
        uint32_t in1_index_subblock_offset = 0;
        for (uint32_t in1_sub = 0; in1_sub < in1_num_subblocks; in1_sub++) {
            matmul_acquire_dst();
            if (enable_reload) {
                matmul_reload_partials<MatmulMode::BLOCK>(cfg, out_subblock_num_tiles);
            }
            matmul_accumulate<MatmulMode::BLOCK>(
                cfg, in0_index_subblock_offset, in1_index_subblock_offset, 0, cfg.kt_dim, 1, in1_block_w, 0);
            if (last_out) {
                matmul_pack_output(cfg.out_cb_id, out_subblock_num_tiles);
            } else {
                matmul_pack_partials(cfg, out_subblock_num_tiles);
            }
            in1_index_subblock_offset += cfg.ct_dim;
        }
        in0_index_subblock_offset += in0_subblock_num_tiles;
    }

    cb_pop_front(cfg.in0_cb_id, in0_block_num_tiles);
    cb_pop_front(cfg.in1_cb_id, in1_block_num_tiles);
}

template <MatmulMode mode>
ALWI void matmul(const MatmulConfig& cfg, const MatmulBlockShape& shape) {
    if constexpr (mode == MatmulMode::TILE) {
        for (uint32_t b = 0; b < shape.batch; b++) {
            for (uint32_t m = 0; m < shape.num_blocks_h; m++) {
                for (uint32_t n = 0; n < shape.num_blocks_w; n++) {
                    matmul_compute_tile(cfg, shape.num_blocks_inner);
                }
            }
        }
    } else {
        for (uint32_t b = 0; b < shape.batch; b++) {
            for (uint32_t bh = 0; bh < shape.num_blocks_h; bh++) {
                for (uint32_t bw = 0; bw < shape.num_blocks_w; bw++) {
                    bool enable_reload = false;
                    for (uint32_t block_inner = 0; block_inner < shape.num_blocks_inner; block_inner++) {
                        bool last_out = (block_inner == shape.num_blocks_inner - 1);
                        matmul_inner_block(
                            cfg,
                            shape.in0_num_subblocks,
                            shape.in1_num_subblocks,
                            shape.in0_block_num_tiles,
                            shape.in1_block_num_tiles,
                            shape.in1_block_w,
                            enable_reload,
                            last_out);
                        if (shape.num_blocks_inner > 1) {
                            enable_reload = true;
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Specialized: Reduce-W via Matmul
// =============================================================================

template <bool reinit_per_tile>
ALWI void matmul_reduce_w(const MatmulConfig& cfg, uint32_t count, uint32_t dst_idx) {
    for (uint32_t w = 0; w < count; ++w) {
        cb_wait_front(cfg.in0_cb_id, 1);
        if constexpr (reinit_per_tile) {
#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cfg.in0_cb_id, cfg.in1_cb_id);
#endif
            matmul_init_short<MatmulMode::TILE>(cfg);
        }
        detail::matmul_single<MatmulMode::TILE>(cfg, 0, 0, dst_idx);
        cb_pop_front(cfg.in0_cb_id, 1);
    }
}

// =============================================================================
// Specialized: Attention Patterns
// =============================================================================

ALWI void matmul_attention(const MatmulConfig& cfg, uint32_t inner_dim, bool progressive_in0) {
    for (uint32_t kt = 0; kt < inner_dim; ++kt) {
        if (progressive_in0) {
            cb_wait_front(cfg.in0_cb_id, kt + 1);
        }
        cb_wait_front(cfg.in1_cb_id, 1);
        detail::matmul_single<MatmulMode::TILE>(cfg, kt, 0, 0);
        cb_pop_front(cfg.in1_cb_id, 1);
    }
}

template <MatmulMode mode>
ALWI void matmul_reduce_subblock_inplace(
    const MatmulConfig& cfg, uint32_t num_subblocks, uint32_t subblock_tiles) {
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

// =============================================================================
// Specialized: MoE Patterns
// =============================================================================

template <MatmulMode mode>
ALWI uint32_t matmul_moe_with_bias(
    const MatmulConfig& cfg,
    const MatmulConfig& bias_cfg,
    uint32_t in0_start,
    uint32_t num_blocks,
    uint32_t tiles_per_block,
    uint32_t tile_stride,
    uint32_t limit) {
    uint32_t k_tracker = 0;
    uint32_t in0_index = in0_start;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_wait_front(cfg.in1_cb_id, tiles_per_block);
        uint32_t last_k_index = 0;
        for (uint32_t k = 0; k < tiles_per_block; k += tile_stride) {
            if (k_tracker == limit) {
                last_k_index = k;
                break;
            }
            detail::matmul_single<mode>(cfg, in0_index, k, 0);
            in0_index++;
            k_tracker++;
        }
        if (k_tracker == limit) {
            detail::matmul_single<mode>(bias_cfg, 0, last_k_index, 0);
        }
        cb_pop_front(cfg.in1_cb_id, tiles_per_block);
    }
    return in0_index;
}

template <MatmulMode mode>
ALWI void matmul_moe_w2_dm1_cycling(
    const MatmulConfig& cfg,
    const MatmulConfig& bias_cfg,
    MoeDm1State& dm1,
    uint32_t num_blocks,
    uint32_t tiles_per_block,
    uint32_t tile_stride,
    uint32_t limit,
    uint32_t dm1_rdy_cb,
    uint32_t tiles_per_step,
    uint32_t num_buffers,
    const uint32_t* dm1_table) {
    uint32_t k_tracker = 0;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_wait_front(cfg.in1_cb_id, tiles_per_block);
        uint32_t last_k_index = 0;
        for (uint32_t k = 0; k < tiles_per_block; k += tile_stride) {
            if (k_tracker == limit) {
                last_k_index = k;
                break;
            }
            if (dm1.tiles_remaining == 0) {
                cb_pop_front(dm1_rdy_cb, 1);
                cb_wait_front(dm1_rdy_cb, 1);
                dm1.tiles_remaining = dm1_table[++dm1.step];
                dm1.buf = (dm1.buf >= num_buffers - 1) ? 0 : dm1.buf + 1;
                dm1.offset = dm1.buf * tiles_per_step;
                dm1.index = dm1.offset;
            }
            dm1.tiles_remaining--;
            detail::matmul_single<mode>(cfg, dm1.index, k, 0);
            dm1.index++;
            k_tracker++;
        }
        if (k_tracker == limit) {
            detail::matmul_single<mode>(bias_cfg, 0, last_k_index, 0);
        }
        cb_pop_front(cfg.in1_cb_id, tiles_per_block);
    }
}

template <MatmulMode mode>
ALWI void matmul_moe_w2_dm1_linear(
    const MatmulConfig& cfg,
    MoeDm1State& dm1,
    uint32_t num_blocks,
    uint32_t tiles_per_block,
    uint32_t tile_stride,
    uint32_t dm1_rdy_cb,
    uint32_t tiles_per_step,
    const uint32_t* dm1_table,
    uint32_t last_block_early_exit_k) {
    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_wait_front(cfg.in1_cb_id, tiles_per_block);
        for (uint32_t k = 0; k < tiles_per_block; k += tile_stride) {
            if ((block == (num_blocks - 1)) && (k == last_block_early_exit_k)) {
                cb_pop_front(dm1_rdy_cb, 1);
                break;
            }
            if (dm1.tiles_remaining == 0) {
                cb_pop_front(dm1_rdy_cb, 1);
                cb_wait_front(dm1_rdy_cb, 1);
                dm1.tiles_remaining = dm1_table[++dm1.step];
                dm1.offset += tiles_per_step;
                dm1.index = dm1.offset;
            }
            dm1.tiles_remaining--;
            detail::matmul_single<mode>(cfg, dm1.index, k, 0);
            dm1.index++;
        }
        cb_pop_front(cfg.in1_cb_id, tiles_per_block);
    }
}

}  // namespace compute_kernel_lib
