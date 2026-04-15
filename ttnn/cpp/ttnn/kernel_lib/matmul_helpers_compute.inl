// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

#ifdef ARCH_BLACKHOLE
#include "api/compute/experimental/matmul_custom.h"
#endif

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

template <MatmulMode mode>
ALWI void matmul_accumulate(
    const MatmulConfig& cfg,
    uint32_t in0_start, uint32_t in1_start, uint32_t dst_start,
    uint32_t count, uint32_t in0_stride, uint32_t in1_stride, uint32_t dst_stride) {
    for (uint32_t k = 0; k < count; ++k) {
        detail::matmul_single<mode>(
            cfg, in0_start + k * in0_stride, in1_start + k * in1_stride, dst_start + k * dst_stride);
    }
}

#ifdef ARCH_BLACKHOLE
template <MatmulMode mode>
ALWI void matmul_accumulate_no_mop(
    const MatmulConfig& cfg,
    uint32_t in0_start, uint32_t in1_start, uint32_t dst_start,
    uint32_t count, uint32_t in0_stride, uint32_t in1_stride, uint32_t dst_stride) {
    static_assert(mode == MatmulMode::BLOCK, "matmul_accumulate_no_mop is only supported in BLOCK mode");
    for (uint32_t k = 0; k < count; ++k) {
        ckernel::matmul_block_no_mop(
            cfg.in0_cb_id, cfg.in1_cb_id,
            in0_start + k * in0_stride, in1_start + k * in1_stride, dst_start + k * dst_stride,
            cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
    }
}
#endif

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
// DST-managed helpers
// =============================================================================

template <MatmulMode mode, typename PostComputeFn>
ALWI void matmul_single_and_pack(
    const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx,
    uint32_t out_cb, PostComputeFn post_compute) {
    // Full lifecycle: init + reconfig + input CB wait + DST + output CB + input CB pop
    matmul_init_short<mode>(cfg);
    reconfig_data_format(cfg.in1_cb_id, cfg.in0_cb_id);
    cb_wait_front(cfg.in0_cb_id, 1);
    cb_wait_front(cfg.in1_cb_id, 1);

    cb_reserve_back(out_cb, 1);
    tile_regs_acquire();
    detail::matmul_single<mode>(cfg, in0_idx, in1_idx, 0);
    post_compute(1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, 1);

    cb_pop_front(cfg.in0_cb_id, 1);
}

template <MatmulMode mode>
ALWI void matmul_reduce_subblock_inplace(
    const MatmulConfig& cfg, uint32_t num_subblocks, uint32_t subblock_tiles, uint32_t total_in0_tiles) {
    // Full lifecycle: init + reconfig + input CB waits + per-subblock DST + output CB
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

// =============================================================================
// SDPA Helpers: Absolute-offset packing patterns
// =============================================================================

template <MatmulMode mode, bool blocked_pack, typename PostPackFn>
ALWI void matmul_and_pack_absolute(
    const MatmulConfig& cfg,
    uint32_t in0_start, uint32_t in1_start,
    uint32_t inner_dim, uint32_t in1_stride,
    uint32_t out_num_cols, uint32_t row_offset, uint32_t col_offset,
    PostPackFn post_pack) {

    const uint32_t subblock_h = cfg.rt_dim;
    const uint32_t subblock_w = cfg.ct_dim;

    tile_regs_acquire();
#ifdef ARCH_BLACKHOLE
    detail::matmul_accumulate_no_mop<mode>(cfg, in0_start, in1_start, 0, inner_dim, 1, in1_stride, 0);
#else
    detail::matmul_accumulate<mode>(cfg, in0_start, in1_start, 0, inner_dim, 1, in1_stride, 0);
#endif
    tile_regs_commit();

    tile_regs_wait();
    uint32_t dst_idx = 0;
#ifdef ARCH_BLACKHOLE
    if constexpr (blocked_pack) {
        for (uint32_t r = 0; r < subblock_h; r++) {
            uint32_t out_row_offset = (r + row_offset) * out_num_cols;
            pack_tile<true>(dst_idx, cfg.out_cb_id, out_row_offset + col_offset);
            dst_idx += subblock_w;
        }
    } else
#endif
    {
        for (uint32_t r = 0; r < subblock_h; r++) {
            uint32_t out_row_offset = (r + row_offset) * out_num_cols;
            for (uint32_t c = 0; c < subblock_w; c++) {
                pack_tile<true>(dst_idx, cfg.out_cb_id, out_row_offset + col_offset + c);
                dst_idx++;
            }
        }
    }
    post_pack();
    tile_regs_release();
}

template <MatmulMode mode, typename PostComputeFn>
ALWI void matmul_blocks_absolute(
    const MatmulConfig& cfg,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t in0_num_subblocks, uint32_t in1_num_subblocks,
    PostComputeFn post_compute) {

    const uint32_t subblock_h = cfg.rt_dim;
    const uint32_t subblock_w = cfg.ct_dim;

    matmul_init_short<mode>(cfg);

    const uint32_t output_num_tiles = M * N;
    const uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    const uint32_t in0_subblock_all_cols_num_tiles = subblock_h * N;
    const uint32_t in0_subblock_num_tiles = subblock_h * cfg.kt_dim;

    reconfig_data_format(cfg.in1_cb_id, cfg.in0_cb_id);
    cb_wait_front(cfg.in1_cb_id, K * N);
    cb_reserve_back(cfg.out_cb_id, output_num_tiles);

    uint32_t in0_index_offset = 0;
    uint32_t in0_wait_tiles = in0_subblock_num_tiles;

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        cb_wait_front(cfg.in0_cb_id, in0_wait_tiles);
        uint32_t in1_index_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();
            detail::matmul_accumulate<mode>(cfg, in0_index_offset, in1_index_offset, 0, cfg.kt_dim, 1, N, 0);

            post_compute(out_subblock_num_tiles);

            tile_regs_commit();
            tile_regs_wait();

            uint32_t dst_idx = 0;
            uint32_t out_col_offset = in1_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                uint32_t out_row_offset = r * N;
                for (uint32_t c = 0; c < subblock_w; c++) {
                    pack_tile<true>(dst_idx, cfg.out_cb_id, out_row_offset + out_col_offset + c);
                    dst_idx++;
                }
            }
            tile_regs_release();

            in1_index_offset += subblock_w;
        }
        in0_index_offset += in0_subblock_num_tiles;
        in0_wait_tiles += in0_subblock_num_tiles;
        cb_push_back(cfg.out_cb_id, in0_subblock_all_cols_num_tiles);
    }

    cb_pop_front(cfg.in1_cb_id, K * N);
}

}  // namespace compute_kernel_lib
