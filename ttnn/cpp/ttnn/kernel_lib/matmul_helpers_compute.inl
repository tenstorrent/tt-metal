// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

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
// Internal: single matmul dispatch (tile vs block)
// =============================================================================

namespace detail {

template <MatmulMode mode>
ALWI void matmul_single(const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t dst_idx) {
    if constexpr (mode == MatmulMode::BLOCK) {
        ckernel::matmul_block(
            cfg.in0_cb_id,
            cfg.in1_cb_id,
            in0_idx,
            in1_idx,
            dst_idx,
            cfg.transpose,
            cfg.ct_dim,
            cfg.rt_dim,
            cfg.kt_dim);
    } else {
        ckernel::matmul_tiles(cfg.in0_cb_id, cfg.in1_cb_id, in0_idx, in1_idx, dst_idx);
    }
}

}  // namespace detail

// =============================================================================
// Layer 0: Initialization
// =============================================================================

template <MatmulMode mode>
ALWI void matmul_init(const MatmulConfig& cfg) {
    if constexpr (mode == MatmulMode::BLOCK) {
        ckernel::mm_block_init(
            cfg.in0_cb_id, cfg.in1_cb_id, cfg.out_cb_id, cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
    } else {
        ckernel::mm_init(cfg.in0_cb_id, cfg.in1_cb_id, cfg.out_cb_id, cfg.transpose);
    }
}

template <MatmulMode mode>
ALWI void matmul_init_short(const MatmulConfig& cfg) {
    if constexpr (mode == MatmulMode::BLOCK) {
        ckernel::mm_block_init_short(
            cfg.in0_cb_id, cfg.in1_cb_id, cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
    } else {
        ckernel::mm_init_short(cfg.in0_cb_id, cfg.in1_cb_id, cfg.transpose);
    }
}

template <MatmulMode mode>
ALWI void matmul_init_short_with_dt(const MatmulConfig& cfg, uint32_t old_in1_cb_id) {
    if constexpr (mode == MatmulMode::BLOCK) {
        ckernel::mm_block_init_short_with_dt(
            cfg.in0_cb_id, cfg.in1_cb_id, old_in1_cb_id, cfg.transpose, cfg.ct_dim, cfg.rt_dim, cfg.kt_dim);
    } else {
        ckernel::mm_init_short_with_dt(cfg.in0_cb_id, cfg.in1_cb_id, old_in1_cb_id, cfg.transpose);
    }
}

template <MatmulMode mode>
ALWI void matmul_init_short_with_both_dt(const MatmulConfig& cfg, uint32_t old_in0_cb_id, uint32_t old_in1_cb_id) {
    static_assert(mode == MatmulMode::BLOCK, "matmul_init_short_with_both_dt is only supported in BLOCK mode");
    ckernel::mm_block_init_short_with_both_dt(
        cfg.in0_cb_id,
        cfg.in1_cb_id,
        old_in0_cb_id,
        old_in1_cb_id,
        cfg.transpose,
        cfg.ct_dim,
        cfg.rt_dim,
        cfg.kt_dim);
}

// =============================================================================
// detail:: building blocks — see warning in .hpp
// =============================================================================

template <MatmulMode mode>
ALWI void detail::matmul_accumulate(
    const MatmulConfig& cfg,
    uint32_t in0_start, uint32_t in1_start, uint32_t dst_start,
    uint32_t count, uint32_t in0_stride, uint32_t in1_stride, uint32_t dst_stride) {
    for (uint32_t k = 0; k < count; ++k) {
        detail::matmul_single<mode>(cfg, in0_start + k * in0_stride, in1_start + k * in1_stride, dst_start + k * dst_stride);
    }
}

template <MatmulMode mode>
ALWI void detail::matmul_accumulate_subblock(
    const MatmulConfig& cfg,
    uint32_t in0_subblock_offset, uint32_t in1_subblock_offset,
    uint32_t out_h, uint32_t out_w, uint32_t inner_dim, uint32_t in1_stride) {
    uint32_t dst_index = 0;
    for (uint32_t h = 0; h < out_h; ++h) {
        for (uint32_t w = 0; w < out_w; ++w) {
            detail::matmul_accumulate<mode>(
                cfg, in0_subblock_offset + h * inner_dim, in1_subblock_offset + w, dst_index, inner_dim, 1, in1_stride, 0);
            ++dst_index;
        }
    }
}

#ifdef ARCH_BLACKHOLE
template <MatmulMode mode>
ALWI void detail::matmul_accumulate_no_mop(
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

ALWI void detail::matmul_pack_to_cb(uint32_t dest_cb_id, uint32_t num_tiles) {
    tile_regs_commit();
    cb_reserve_back(dest_cb_id, num_tiles);
    tile_regs_wait();
    for (uint32_t i = 0; i < num_tiles; i++) {
        pack_tile(i, dest_cb_id);
    }
    tile_regs_release();
    cb_push_back(dest_cb_id, num_tiles);
}

ALWI void detail::matmul_pack_to_partials(const MatmulConfig& cfg, uint32_t num_tiles) {
    detail::matmul_pack_to_cb(cfg.partials_cb_id, num_tiles);
}

template <MatmulMode mode>
ALWI void detail::matmul_reload_partials(const MatmulConfig& cfg, uint32_t num_tiles) {
    copy_tile_to_dst_init_short_with_dt(cfg.in1_cb_id, cfg.partials_cb_id);
    cb_wait_front(cfg.partials_cb_id, num_tiles);
    copy_block_matmul_partials(cfg.partials_cb_id, 0, 0, num_tiles);
    cb_pop_front(cfg.partials_cb_id, num_tiles);
    matmul_init_short_with_dt<mode>(cfg, cfg.partials_cb_id);
}

// =============================================================================
// Layer 4: Compound patterns
// =============================================================================

template <MatmulMode mode, typename PostComputeFn>
ALWI void matmul_accumulate_and_pack(
    const MatmulConfig& cfg,
    uint32_t in0_index_start,
    uint32_t in1_index_start,
    uint32_t inner_dim,
    uint32_t in1_stride,
    uint32_t dest_cb_id,
    uint32_t num_tiles,
    bool reload,
    PostComputeFn post_compute) {
    tile_regs_acquire();
    if (reload) {
        detail::matmul_reload_partials<mode>(cfg, num_tiles);
    }
    detail::matmul_accumulate<mode>(cfg, in0_index_start, in1_index_start, 0, inner_dim, 1, in1_stride, 0);
    post_compute(num_tiles);
    detail::matmul_pack_to_cb(dest_cb_id, num_tiles);
}

template <MatmulMode mode>
ALWI void matmul_compute_one_tile(const MatmulConfig& cfg, uint32_t inner_dim) {
    tile_regs_acquire();
    for (uint32_t k = 0; k < inner_dim; ++k) {
        cb_wait_front(cfg.in0_cb_id, 1);
        cb_wait_front(cfg.in1_cb_id, 1);
        detail::matmul_single<mode>(cfg, 0, 0, 0);
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

template <MatmulMode mode>
ALWI void matmul_compute_inner_block(
    const MatmulConfig& cfg,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in1_block_num_tiles,
    uint32_t in1_block_w,
    bool enable_reload,
    bool last_out) {
    static_assert(mode == MatmulMode::BLOCK, "matmul_compute_inner_block is only supported in BLOCK mode");
    uint32_t out_subblock_num_tiles = cfg.ct_dim * cfg.rt_dim;
    uint32_t in0_subblock_num_tiles = cfg.rt_dim * cfg.kt_dim;

    cb_wait_front(cfg.in0_cb_id, in0_block_num_tiles);
    cb_wait_front(cfg.in1_cb_id, in1_block_num_tiles);

    uint32_t in0_index_subblock_offset = 0;
    for (uint32_t in0_sub = 0; in0_sub < in0_num_subblocks; in0_sub++) {
        uint32_t in1_index_subblock_offset = 0;
        for (uint32_t in1_sub = 0; in1_sub < in1_num_subblocks; in1_sub++) {
            tile_regs_acquire();
            if (enable_reload) {
                detail::matmul_reload_partials<mode>(cfg, out_subblock_num_tiles);
            }
            detail::matmul_accumulate<mode>(cfg, in0_index_subblock_offset, in1_index_subblock_offset, 0, cfg.kt_dim, 1, in1_block_w, 0);
            if (last_out) {
                detail::matmul_pack_to_cb(cfg.out_cb_id, out_subblock_num_tiles);
            } else {
                detail::matmul_pack_to_partials(cfg, out_subblock_num_tiles);
            }
            in1_index_subblock_offset += cfg.ct_dim;
        }
        in0_index_subblock_offset += in0_subblock_num_tiles;
    }

    cb_pop_front(cfg.in0_cb_id, in0_block_num_tiles);
    cb_pop_front(cfg.in1_cb_id, in1_block_num_tiles);
}

// =============================================================================
// Layer 5: Specialized patterns
// =============================================================================

template <MatmulMode mode>
ALWI void detail::matmul_reduce_w(const MatmulConfig& cfg, uint32_t count, uint32_t dst_idx) {
    for (uint32_t w = 0; w < count; ++w) {
        cb_wait_front(cfg.in0_cb_id, 1);
        detail::matmul_single<mode>(cfg, 0, 0, dst_idx);
        cb_pop_front(cfg.in0_cb_id, 1);
    }
}

template <MatmulMode mode>
ALWI void detail::matmul_reduce_w_with_init(const MatmulConfig& cfg, uint32_t count, uint32_t dst_idx) {
    for (uint32_t w = 0; w < count; ++w) {
        cb_wait_front(cfg.in0_cb_id, 1);
#if defined FP32_DEST_ACC_EN
        reconfig_data_format(cfg.in0_cb_id, cfg.in1_cb_id);
#endif
        matmul_init_short<mode>(cfg);
        detail::matmul_single<mode>(cfg, 0, 0, dst_idx);
        cb_pop_front(cfg.in0_cb_id, 1);
    }
}

template <MatmulMode mode>
ALWI void matmul_reduce_w_and_pack(const MatmulConfig& cfg, uint32_t count, uint32_t dst_idx, uint32_t out_cb) {
    tile_regs_acquire();
    detail::matmul_reduce_w<mode>(cfg, count, dst_idx);
    tile_regs_commit();
    cb_reserve_back(out_cb, 1);
    tile_regs_wait();
    pack_tile(dst_idx, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, 1);
}

template <MatmulMode mode>
ALWI void matmul_reduce_subblock_inplace(const MatmulConfig& cfg, uint32_t num_subblocks, uint32_t subblock_tiles) {
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
// Layer 6: Full automated matmul
// =============================================================================

template <MatmulMode mode>
ALWI void matmul(const MatmulConfig& cfg, const MatmulBlockShape& shape) {
    if constexpr (mode == MatmulMode::TILE) {
        for (uint32_t b = 0; b < shape.batch; b++) {
            for (uint32_t m = 0; m < shape.num_blocks_h; m++) {
                for (uint32_t n = 0; n < shape.num_blocks_w; n++) {
                    matmul_compute_one_tile<mode>(cfg, shape.num_blocks_inner);
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
                        matmul_compute_inner_block<mode>(
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
// Layer 7: Single-tile matmul with DST+CB encapsulation
// =============================================================================

template <MatmulMode mode, typename PostComputeFn>
ALWI void matmul_single_and_pack(
    const MatmulConfig& cfg,
    uint32_t in0_idx,
    uint32_t in1_idx,
    uint32_t out_cb,
    PostComputeFn post_compute) {

    cb_reserve_back(out_cb, 1);
    tile_regs_acquire();
    detail::matmul_single<mode>(cfg, in0_idx, in1_idx, 0);
    post_compute(1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, 1);
}

// =============================================================================
// SDPA Helpers: Absolute-offset packing patterns
// =============================================================================

template <MatmulMode mode, bool blocked_pack, typename PostPackFn>
ALWI void matmul_and_pack_absolute(
    const MatmulConfig& cfg,
    uint32_t in0_start,
    uint32_t in1_start,
    uint32_t inner_dim,
    uint32_t in1_stride,
    uint32_t out_num_cols,
    uint32_t row_offset,
    uint32_t col_offset,
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
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
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
