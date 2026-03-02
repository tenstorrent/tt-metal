// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Streaming SDPA compute helpers.
// Included only by sdpa.cpp when use_streaming_compute is true.
// Depends on primitives from compute_common.hpp (must be included first).

#pragma once

#include <type_traits>

#include "tools/profiler/kernel_profiler.hpp"
#ifdef ARCH_BLACKHOLE
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#endif

// Template-driven profiling: MaybeDeviceZoneScopedN(ENABLED, name)
// When ENABLED=true: RAII profileScope writes timestamps (same as DeviceZoneScopedN)
// When ENABLED=false: empty struct, zero overhead (compiler eliminates entirely)
#if defined(PROFILE_KERNEL)
template <bool Enabled, uint32_t timer_id>
struct MaybeProfileScope {
    inline __attribute__((always_inline)) MaybeProfileScope() {}
    inline __attribute__((always_inline)) ~MaybeProfileScope() {}
};
template <uint32_t timer_id>
struct MaybeProfileScope<true, timer_id> : kernel_profiler::profileScope<timer_id> {};

#define MaybeDeviceZoneScopedN(ENABLED, name)                                  \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    MaybeProfileScope<ENABLED, hash> zone;
#else
#define MaybeDeviceZoneScopedN(ENABLED, name)
#endif

/**
 * Push tiles to make them visible for UNPACK reads, but rewind wr_ptr so
 * subsequent pack_tile<true> offsets remain relative to a stable base.
 * This eliminates the need for separate row buffers.
 */
ALWI void cb_push_back_hold_wr_ptr(uint32_t cb_id, uint32_t num_tiles) {
    cb_push_back(cb_id, num_tiles);
    PACK(({
        auto& intf = get_local_cb_interface(cb_id);
        intf.fifo_wr_ptr -= num_tiles * intf.fifo_page_size;
        uint32_t fifo_start = intf.fifo_limit - intf.fifo_size;
        if (intf.fifo_wr_ptr < fifo_start) {
            intf.fifo_wr_ptr += intf.fifo_size;
        }
    }));
}

/**
 * Blocked subblock matmul with absolute offset packing.
 * Always uses pack_tile<true> at row-major positions in out_cb.
 */
template <
    bool TRANSPOSE,
    uint32_t SUBBLOCK_W,
    uint32_t SUBBLOCK_H,
    uint32_t INNER_DIM,
    uint32_t IN1_STRIDE,
    uint32_t OUT_NUM_COLS,
    bool blocked_pack = false>
void blocked_matmul_and_pack(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t in0_index_start,
    uint32_t in1_index_start,
    uint32_t q_subblock,
    uint32_t out_col_offset) {
    tile_regs_acquire();
    uint32_t dst_index = 0;
    uint32_t in0_index = in0_index_start;
    uint32_t in1_index = in1_index_start;
    for (uint32_t inner = 0; inner < INNER_DIM; ++inner) {
#ifdef ARCH_BLACKHOLE
        matmul_block_no_mop(
            in0_cb, in1_cb, in0_index, in1_index, dst_index, TRANSPOSE, SUBBLOCK_W, SUBBLOCK_H, INNER_DIM);
#else
        matmul_block(in0_cb, in1_cb, in0_index, in1_index, dst_index, TRANSPOSE, SUBBLOCK_W, SUBBLOCK_H, INNER_DIM);
#endif
        in0_index++;
        in1_index += IN1_STRIDE;
    }
    tile_regs_commit();

    tile_regs_wait();
    uint32_t dst_idx = 0;
#ifdef ARCH_BLACKHOLE
    if constexpr (blocked_pack) {
        for (uint32_t r = 0; r < SUBBLOCK_H; r++) {
            uint32_t out_row_offset = (r + q_subblock * SUBBLOCK_H) * OUT_NUM_COLS;
            pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset);
            dst_idx += SUBBLOCK_W;
        }
    } else
#endif
    {
        for (uint32_t r = 0; r < SUBBLOCK_H; r++) {
            uint32_t out_row_offset = (r + q_subblock * SUBBLOCK_H) * OUT_NUM_COLS;
            for (uint32_t c = 0; c < SUBBLOCK_W; c++) {
                pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c);
                dst_idx++;
            }
        }
    }
    tile_regs_release();
}

/**
 * Per-row-group max reduction with optional eltwise_max against prev values.
 * Reads from in0_cb at row group offset, writes to out_cb sequentially.
 */
template <PoolType pool_type, ReduceDim reduce_dim, uint32_t scale_cb, uint32_t cols, uint32_t SBH>
void reduce_c_row_group(
    uint32_t in0_cb, uint32_t out_cb, uint32_t prev_cb, uint32_t row_group_index, bool do_eltwise_max = false) {
    constexpr uint32_t GROUP_SIZE = SBH;
    const uint32_t row_start = row_group_index * GROUP_SIZE;

    const uint32_t cumulative_input_tiles = (row_group_index + 1) * GROUP_SIZE * cols;
    const uint32_t cumulative_prev_tiles = (row_group_index + 1) * GROUP_SIZE;

    // scale_cb assumed ready (waited once at kernel init)
    // cb_wait_front(scale_cb, 1);

    tile_regs_acquire();

    if (do_eltwise_max) {
        cb_wait_front(prev_cb, cumulative_prev_tiles);
        sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
        for (uint32_t i = 0; i < GROUP_SIZE; i++) {
            copy_tile(prev_cb, row_start + i, i);
        }
    }

    // Deferred: wait for in0_cb just before its first use (reduce_block_max_row).
    // When do_eltwise_max=true, the prev_cb wait + copy_tile work above can overlap
    // with in0_cb data arrival.
    cb_wait_front(in0_cb, cumulative_input_tiles);

    reduce_block_max_row_init<cols>();
    for (uint32_t i = 0; i < GROUP_SIZE; i++) {
        const uint32_t input_tile_start = (row_start + i) * cols;
        reduce_block_max_row<cols>(in0_cb, scale_cb, input_tile_start, i);
    }
    reduce_block_max_row_uninit(in0_cb);

    tile_regs_commit();
    tile_regs_wait();

    for (uint32_t i = 0; i < GROUP_SIZE; i++) {
        pack_tile<false>(i, out_cb);
    }

    tile_regs_release();
}

/**
 * In-place sub_exp on cb_qkt_im: subtracts max, applies exp with ReLU clamping,
 * writes back to same positions. Accumulates row sums into reduce_cb.
 */
template <
    bool PROFILING_ENABLED,
    uint32_t scale_fp32,
    uint32_t SBH,
    uint32_t SBW,
    bool do_reduce,
    int vector_mode = (int)VectorMode::RC,
    bool blocked_pack = false>
void sub_exp_block_bcast_cols(
    uint32_t inout_cb,
    uint32_t max_cb,
    uint32_t reduce_cb,
    uint32_t cols_in_row,
    uint32_t q_subblock,
    uint32_t kt_subblock) {
    constexpr uint32_t tiles_per_row = SBH;
    constexpr uint32_t tiles_per_column = SBW;
    static_assert(tiles_per_row * tiles_per_column <= 8, "SBH * SBW must fit in DST (max 8 tiles)");
    const uint32_t max_row_base = q_subblock * tiles_per_row;
    const uint32_t global_col_base = kt_subblock * tiles_per_column;

    {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "SUB_EXP_BLOCK_INIT");
#ifdef ARCH_BLACKHOLE
        sub_bcast_cols_init_short_custom(inout_cb, max_cb, tiles_per_column);
#else
        sub_bcast_cols_init_short(inout_cb, max_cb);
#endif
        PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
    }

    // inout_cb assumed ready (max_cb was already computed from it)
    // cb_wait_front(inout_cb, (q_subblock + 1) * tiles_per_row * cols_in_row);
    cb_wait_front(max_cb, (q_subblock + 1) * tiles_per_row);

    tile_regs_acquire();
    {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "SUB");
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
#ifdef ARCH_BLACKHOLE
            uint32_t in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base;
            sub_tiles_bcast_cols_custom(
                inout_cb, max_cb, in0_tile_index, max_row_base + i, dst_index, tiles_per_column);
            dst_index += tiles_per_column;
#else
            for (uint32_t j = 0; j < tiles_per_column; j++) {
                // Absolute position in cb_qkt_im
                uint32_t in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base + j;
                sub_tiles_bcast_cols(inout_cb, max_cb, in0_tile_index, max_row_base + i, dst_index++);
            }
#endif
        }
    }
    tile_regs_commit();

    tile_regs_wait();
    {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "EXP");
        uint32_t dst_index = 0;
        constexpr int iterations = (vector_mode == (int)VectorMode::RC) ? 32 : 8;
        constexpr int vector_mode_exp = (vector_mode == (int)VectorMode::RC) ? (int)VectorMode::None : vector_mode;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < tiles_per_column; j++) {
                exp_packthread_tile<true, true, false, false, InputClamping::None, iterations>(
                    dst_index++, vector_mode_exp);
            }
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    }

    {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "PACK SUB_EXP");
        // Pack back to inout_cb at the same absolute positions
        uint32_t dst_index = 0;
#ifdef ARCH_BLACKHOLE
        if constexpr (blocked_pack) {
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                uint32_t in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base;
                pack_tile<true>(dst_index, inout_cb, in0_tile_index);
                dst_index += tiles_per_column;
            }
        } else
#endif
        {
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                for (uint32_t j = 0; j < tiles_per_column; ++j) {
                    uint32_t in0_tile_index = (max_row_base + i) * cols_in_row + global_col_base + j;
                    pack_tile<true>(dst_index++, inout_cb, in0_tile_index);
                }
            }
        }

        // Reduce to reduce_cb: first tile of first kt_subblock overwrites, rest accumulate
        if constexpr (do_reduce) {
#ifdef ARCH_BLACKHOLE
            if constexpr (blocked_pack) {
                PACK((llk_pack_mop_config<false, false, false>(reduce_cb, 1)));
            }
#endif
            dst_index = 0;
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                if (global_col_base > 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                } else {
                    PACK((llk_pack_reconfig_l1_acc(0)));
                }
                for (uint32_t j = 0; j < tiles_per_column; ++j) {
                    pack_tile<true>(dst_index++, reduce_cb, max_row_base + i);
                    if (global_col_base == 0 && j == 0) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
            }
        }
    }

    tile_regs_release();

    // Restore packer ReLU config after all exp operations complete
    PACK((llk_pack_relu_config(ReluType::NO_RELU)));
    if constexpr (do_reduce) {
#ifdef ARCH_BLACKHOLE
        if constexpr (blocked_pack) {
            PACK((llk_pack_mop_config<false, false, false>(reduce_cb, tiles_per_column)));
        }
#endif
        PACK((llk_pack_reconfig_l1_acc(0)));
    }
}

/**
 * Column-only exp(prev_max - cur_max) for SALAD corrections.
 * Operates on first-column subset of tiles.
 */
template <bool PROFILING_ENABLED, uint32_t scale_fp32, uint32_t SBH>
void sub_exp_first_col_blocks(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock) {
    constexpr uint32_t tiles_per_row = SBH;
    const uint32_t global_row_base = q_subblock * tiles_per_row;
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    sub_tiles_init(in0_cb, in1_cb);

    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    {
        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            uint32_t tile_index = global_row_base + i;
            sub_tiles(in0_cb, in1_cb, tile_index, tile_index, i);
        }
        tile_regs_commit();
    }

    {
        tile_regs_wait();
        for (uint32_t dst_index = 0; dst_index < tiles_per_row; dst_index++) {
            PACK((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(dst_index)));
        }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));

        for (uint32_t i = 0; i < tiles_per_row; i++) {
            pack_tile<false>(i, out_cb);
        }

        tile_regs_release();
    }
}

/**
 * SALAD sum correction: out_cb[row] += in0_cb[row] * bcast_cols(in1_cb[row])
 * with L1 accumulate at explicit offset.
 */
template <uint32_t SBH>
void mul_bcast_cols_l1_acc(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock, uint32_t write_q_subblock) {
    constexpr uint32_t tiles_per_row = SBH;
    const uint32_t read_row_base = q_subblock * tiles_per_row;
    const uint32_t write_row_base = write_q_subblock * tiles_per_row;

    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    PACK((llk_pack_reconfig_l1_acc(1)));
    tile_regs_acquire();
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        uint32_t src_tile_index = read_row_base + i;
        mul_tiles_bcast_cols(in0_cb, in1_cb, src_tile_index, src_tile_index, i);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        pack_tile<true>(i, out_cb, write_row_base + i);
    }
    tile_regs_release();
    PACK((llk_pack_reconfig_l1_acc(0)));
}

/**
 * SALAD output correction: block multiply with bcast_cols and L1 accumulate.
 */
template <uint32_t SBH, uint32_t SBW>
void mul_block_bcast_cols_acc(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock, uint32_t write_q_subblock) {
    constexpr uint32_t tiles_per_row = SBH;
    constexpr uint32_t tiles_per_column = SBW;
    static_assert(tiles_per_row * tiles_per_column <= 8, "SBH * SBW must fit in DST (max 8 tiles)");
    const uint32_t read_row_base = q_subblock * tiles_per_row;
    const uint32_t write_row_base = write_q_subblock * tiles_per_row;
    mul_bcast_cols_init_short(in0_cb, in1_cb);

    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row * tiles_per_column);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    PACK((llk_pack_reconfig_l1_acc(1)));
    tile_regs_acquire();
    uint32_t dst_index = 0;
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        for (uint32_t j = 0; j < tiles_per_column; j++) {
            uint32_t in0_tile_index = (read_row_base + i) * tiles_per_column + j;
            mul_tiles_bcast_cols(in0_cb, in1_cb, in0_tile_index, read_row_base + i, dst_index++);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    dst_index = 0;
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        for (uint32_t j = 0; j < tiles_per_column; j++) {
            uint32_t out_tile_index = (write_row_base + i) * tiles_per_column + j;
            pack_tile<true>(dst_index++, out_cb, out_tile_index);
        }
    }
    tile_regs_release();
    PACK((llk_pack_reconfig_l1_acc(false)));
}

/**
 * Lightweight padded-K mask: L1-accumulate a single permanently-fronted -inf tile onto padded
 * tile positions in cb_qkt_im. No full mask matrix needed — just 1 tile in neginf_cb.
 *
 * Ported from sdpa_benchmark branch's apply_padded_mask.
 *
 * @tparam num_padded  Number of padded K tiles per row (must be > 0)
 * @tparam num_cols    Total K tiles per row (Sk_chunk_t)
 * @tparam SBH         Sub-rows per row group (subblock_h)
 * @tparam DST_BATCH   Max tiles per DST batch (8 for fp16b half-sync)
 */
template <bool PROFILING_ENABLED, uint32_t num_padded, uint32_t num_cols, uint32_t SBH = 1, uint32_t DST_BATCH = 8>
void apply_padded_mask_lightweight(uint32_t neginf_cb, uint32_t out_cb, uint32_t q_subblock = 0) {
    MaybeDeviceZoneScopedN(PROFILING_ENABLED, "PAD_MASK");
    static_assert(num_padded > 0, "num_padded must be > 0");
    static_assert(num_padded < num_cols, "num_padded must be less than num_cols");
    constexpr uint32_t start = num_cols - num_padded;

    copy_tile_to_dst_init_short(neginf_cb);
    cb_wait_front(neginf_cb, 1);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < SBH; row++) {
        uint32_t row_offset = (q_subblock * SBH + row) * num_cols;
        for (uint32_t base = start; base < num_cols; base += DST_BATCH) {
            uint32_t batch = (num_cols - base < DST_BATCH) ? (num_cols - base) : DST_BATCH;
            tile_regs_acquire();
            for (uint32_t i = 0; i < batch; i++) {
                copy_tile(neginf_cb, 0, i);  // Always tile 0 — single -inf tile
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < batch; i++) {
                pack_tile<true>(i, out_cb, row_offset + base + i);
            }
            tile_regs_release();
        }
    }

    PACK((llk_pack_reconfig_l1_acc(0)));
}

/**
 * Per-row streaming normalization: matmul_reduce + recip-in-DST + mul_bcast_cols.
 * Consumes (pops) sum and output tiles, writes normalized output.
 * scratch_cb is a 1-tile CB reused for the reciprocal intermediate.
 */
template <bool PROFILING_ENABLED, uint32_t SBH, uint32_t head_dim_t_>
void normalize_row_streaming(
    uint32_t cur_sum_cb,
    uint32_t cur_out_cb,
    uint32_t col_identity_cb,
    uint32_t scratch_cb,
    uint32_t normalized_out_cb) {
    for (uint32_t s = 0; s < SBH; s++) {
        // 1+2. Fused matmul_reduce + recip: sum × col_identity → recip → 1/sum in scratch
        {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "NORM_MATMUL_RECIP");
            constexpr uint32_t N = 1;
            mm_block_init_short(cur_sum_cb, col_identity_cb, 0, N, 1, N);
            reconfig_data_format(col_identity_cb, cur_sum_cb);

            cb_wait_front(col_identity_cb, N);
            cb_wait_front(cur_sum_cb, 1);

            cb_reserve_back(scratch_cb, 1);
            tile_regs_acquire();
            matmul_block(cur_sum_cb, col_identity_cb, 0, 0, 0, 0, N, 1, N);
            recip_tile_init();
            MATH((recip_tile_first_column(0)));
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, scratch_cb);
            tile_regs_release();
            cb_push_back(scratch_cb, 1);

            cb_pop_front(cur_sum_cb, 1);
        }

        // 3. Normalize: multiply output tiles by bcast_cols(1/sum)
        static_assert(head_dim_t_ <= 8, "head_dim_t must fit in DST (max 8 tiles)");
        {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "NORM_MUL_BCAST");
            mul_bcast_cols_init_short(cur_out_cb, scratch_cb);
            cb_wait_front(cur_out_cb, head_dim_t_);
            cb_wait_front(scratch_cb, 1);

            cb_reserve_back(normalized_out_cb, head_dim_t_);
            tile_regs_acquire();
            for (uint32_t j = 0; j < head_dim_t_; ++j) {
                mul_tiles_bcast_cols(cur_out_cb, scratch_cb, j, 0, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < head_dim_t_; ++j) {
                pack_tile(j, normalized_out_cb);
            }
            tile_regs_release();
            cb_push_back(normalized_out_cb, head_dim_t_);

            cb_pop_front(scratch_cb, 1);
            cb_pop_front(cur_out_cb, head_dim_t_);
        }
    }
}

// ===================== Streaming SDPA Core Functions =====================

/**
 * L1-accumulate ring mask tiles onto QKT for one row group (q_subblock).
 * Copies mask tiles from cb_mask_in and adds them in-place to cb_qkt_im using L1 accumulation.
 */
template <uint32_t Sk_chunk_t, uint32_t sbh, uint32_t cb_mask_in, uint32_t cb_qkt_im>
void apply_ring_mask_to_qkt(uint32_t q_subblock) {
    constexpr uint32_t DST_BATCH = 8;
    for (uint32_t row = 0; row < sbh; row++) {
        uint32_t out_row_offset = (q_subblock * sbh + row) * Sk_chunk_t;
        uint32_t mask_row_offset = (q_subblock * sbh + row) * Sk_chunk_t;
        copy_tile_to_dst_init_short(cb_mask_in);
        PACK((llk_pack_reconfig_l1_acc(1)));
        for (uint32_t base = 0; base < Sk_chunk_t; base += DST_BATCH) {
            uint32_t batch = (Sk_chunk_t - base < DST_BATCH) ? (Sk_chunk_t - base) : DST_BATCH;
            tile_regs_acquire();
            for (uint32_t i = 0; i < batch; i++) {
                copy_tile(cb_mask_in, mask_row_offset + base + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < batch; i++) {
                pack_tile<true>(i, cb_qkt_im, out_row_offset + base + i);
            }
            tile_regs_release();
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
    }
}

/**
 * One K-chunk iteration of the streaming SDPA algorithm (v2 — no row buffers).
 * Phase 1: Q@KT directly into cb_qkt_im with cb_push_back_hold_wr_ptr, in-place sub_exp.
 * Phase 2: Drain + QKT@V with SALAD corrections, streaming normalization on last K iter.
 */
template <
    bool PROFILING_ENABLED,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t Skt,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t subblock_h,
    bool use_padded_mask,
    bool ring_mode = false,
    bool use_ring_mask = false,
    uint32_t cb_q_in = 0,
    uint32_t cb_kt_in = 0,
    uint32_t cb_v_in = 0,
    uint32_t cb_qkt_im = 0,
    uint32_t cb_identity_scale_in = 0,
    uint32_t cb_exp_max_diff = 0,
    uint32_t cb_col_identity = 0,
    uint32_t cb_recip_scratch = 0,
    uint32_t cb_normalized_out = 0,
    uint32_t cb_mask_in = 0>
void sdpa_inner_loop_step(
    const uint32_t prev_max,
    const uint32_t cur_max,
    const uint32_t prev_sum,
    const uint32_t cur_sum,
    const uint32_t prev_out,
    const uint32_t cur_out,
    const bool is_last_iter,
    const bool is_first_iter,
    [[maybe_unused]] const bool apply_mask = false) {
    constexpr uint32_t sbh = subblock_h;
    constexpr uint32_t in0_block_w = DHt;
    constexpr uint32_t qkt_subblock_w = 8 / sbh;
    // Compute padded K tiles from compile-time Skt and Sk_chunk_t
    constexpr uint32_t padded_k_tiles = (Sk_chunk_t - (Skt % Sk_chunk_t)) % Sk_chunk_t;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / sbh;
    constexpr uint32_t kt_num_subblocks = Sk_chunk_t / qkt_subblock_w;
    constexpr uint32_t q_subblock_num_tiles = sbh * in0_block_w;
    constexpr uint32_t row_tiles = sbh * Sk_chunk_t;

    static_assert(sbh * qkt_subblock_w <= 8, "sbh * qkt_subblock_w must fit in DST (max 8 tiles)");
    static_assert(Sk_chunk_t % qkt_subblock_w == 0, "Sk_chunk_t must be divisible by qkt_subblock_w");
    static_assert(Sq_chunk_t % sbh == 0, "Sq_chunk_t must be divisible by subblock_h");
    static_assert(!(use_padded_mask && use_ring_mask), "use_padded_mask and use_ring_mask are mutually exclusive");

    uint32_t pushed_rows = 0;
    uint32_t q_wait_tiles = q_subblock_num_tiles;
    uint32_t q_index_offset = 0;
    uint32_t kt_index_offset = 0;

    exp_packthread_tile_init<true, true, scale_fp32, InputClamping::None>();

    pack_reconfig_data_format(cb_qkt_im);
    reconfig_data_format(cb_kt_in, cb_q_in);
    cb_reserve_back(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);

    cb_reserve_back(cur_sum, Sq_chunk_t);

    // ========== PHASE 1: Q@KT directly into cb_qkt_im ==========
    // All matmul output goes to cb_qkt_im at absolute offsets via pack_tile<true>.
    // cb_push_back_hold_wr_ptr makes each row visible to UNPACK without advancing wr_ptr.
    PACK((llk_pack_mop_config<false, false, false>(cb_qkt_im, qkt_subblock_w)));
    for (uint32_t q_subblock = 0; q_subblock < q_num_subblocks; q_subblock++) {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Softmax(Q@KT)");
        cb_wait_front(cb_q_in, q_wait_tiles);
        // Deferred KT wait: reader pushes Q before KT, so waiting for Q first
        // allows reserves + MOP config + Q wait to overlap with KT DMA.
        if (q_subblock == 0) {
            cb_wait_front(cb_kt_in, DHt * Sk_chunk_t);
        }
        kt_index_offset = 0;
#ifdef ARCH_BLACKHOLE
        mm_no_mop_init_short(cb_q_in, cb_kt_in, true, qkt_subblock_w, sbh, in0_block_w);
#else
        mm_block_init_short(cb_q_in, cb_kt_in, true, qkt_subblock_w, sbh, in0_block_w);
#endif
        for (uint32_t kt_subblock = 0; kt_subblock < kt_num_subblocks; ++kt_subblock) {
            if (q_subblock > 0) {
                uint32_t prev_q_subblock = q_subblock - 1;
                sub_exp_block_bcast_cols<
                    PROFILING_ENABLED,
                    scale_fp32,
                    sbh,
                    qkt_subblock_w,
                    true,
                    (int)VectorMode::RC,
                    true /*blocked_pack*/>(cb_qkt_im, cur_max, cur_sum, Sk_chunk_t, prev_q_subblock, kt_subblock);
#ifdef ARCH_BLACKHOLE
                mm_no_mop_reinit_short(cb_q_in, cb_kt_in, true, qkt_subblock_w, sbh, in0_block_w);
#else
                mm_block_init_short(cb_q_in, cb_kt_in, true, qkt_subblock_w, sbh, in0_block_w);
#endif
            }

            {
                MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Q@KT MM+Pack");
                blocked_matmul_and_pack<
                    true,
                    qkt_subblock_w,
                    sbh,
                    in0_block_w,
                    Sk_chunk_t,
                    Sk_chunk_t,
                    true /*blocked_pack*/>(
                    cb_q_in,
                    cb_kt_in,
                    cb_qkt_im,
                    q_index_offset,
                    kt_index_offset,
                    q_subblock,
                    kt_subblock * qkt_subblock_w);
            }
            kt_index_offset += qkt_subblock_w;
        }

        // Apply mask on last K chunk: L1-accumulate single -inf tile onto padded K positions.
        if constexpr (padded_k_tiles > 0) {
            if (is_last_iter) {
                PACK((llk_pack_mop_config<false, false, false>(cb_mask_in, 1)));
                apply_padded_mask_lightweight<PROFILING_ENABLED, padded_k_tiles, Sk_chunk_t, sbh>(
                    cb_mask_in, cb_qkt_im, q_subblock);
            }
        }

        // Ring mask: L1-accumulate full mask tiles onto cb_qkt_im for this row group.
        // Wait deferred to here so reader can overlap mask fetch with Q@KT matmul.
        if constexpr (use_ring_mask) {
            if (apply_mask) {
                if (q_subblock == 0) {
                    cb_wait_front(cb_mask_in, Sq_chunk_t * Sk_chunk_t);
                }
                MaybeDeviceZoneScopedN(PROFILING_ENABLED, "RING_MASK");
                apply_ring_mask_to_qkt<Sk_chunk_t, sbh, cb_mask_in, cb_qkt_im>(q_subblock);
            }
        }

        // Push row (visible for UNPACK reads) but keep wr_ptr stable
        cb_push_back_hold_wr_ptr(cb_qkt_im, row_tiles);

        // Max reduce: reads from cb_qkt_im at q_subblock position
        {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Reduce max");
            cb_reserve_back(cur_max, sbh);
            PACK((llk_pack_mop_config<false, false, false>(cur_max, 1)));
            reduce_c_row_group<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_identity_scale_in, Sk_chunk_t, sbh>(
                cb_qkt_im, cur_max, prev_max, q_subblock, !is_first_iter /*do_eltwise_max*/);
            cb_push_back(cur_max, sbh);
            PACK((llk_pack_mop_config<false, false, false>(cb_qkt_im, qkt_subblock_w)));
        }

        q_index_offset += sbh * in0_block_w;
        q_wait_tiles += q_subblock_num_tiles;
    }

    cb_pop_front(cb_kt_in, DHt * Sk_chunk_t);

    // Q is no longer needed after Phase 1. On the last K chunk, pop early so the
    // reader can start fetching the next Q chunk during Phase 2.
    if (is_last_iter) {
        cb_pop_front(cb_q_in, Sq_chunk_t * DHt);
    }

    // Pop ring mask after all row groups processed
    if constexpr (use_ring_mask) {
        if (apply_mask) {
            cb_pop_front(cb_mask_in, Sq_chunk_t * Sk_chunk_t);
        }
    }

    // ========== PHASE 2: Drain last row + QKT@V + SALAD ==========
    // After Phase 1: all rows are pushed (via hold_wr_ptr) in cb_qkt_im.
    // Rows 0..N-2 are softmax'd in-place; row N-1 has raw matmul output.
    {
        constexpr uint32_t qktv_subblock_h = sbh;
        constexpr uint32_t qktv_subblock_w = (vDHt < 4) ? vDHt : 4;
        constexpr uint32_t qktv_in0_block_w = Sk_chunk_t;
        constexpr uint32_t qktv_q_num_subblocks = Sq_chunk_t / qktv_subblock_h;
        constexpr uint32_t qktv_v_num_subblocks = vDHt / qktv_subblock_w;
        constexpr uint32_t qktv_output_num_tiles = Sq_chunk_t * vDHt;
        constexpr uint32_t qktv_in0_subblock_num_tiles = qktv_subblock_h * qktv_in0_block_w;

        uint32_t qktv_in0_index_offset = 0;
        uint32_t qktv_in0_wait_tiles = qktv_in0_subblock_num_tiles;

        // V wait deferred: don't block here. The sub_exp drain loop below
        // doesn't touch V, so the reader's V DMA can overlap with the drain.
        cb_reserve_back(cur_out, qktv_output_num_tiles);

        // q_subblock 0: drain last row's sub_exp in-place + first QKT@V matmul
        PACK((llk_pack_mop_config<false, false, false>(cb_qkt_im, 1)));
        {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Softmax(Q@KT)@V");
            static_assert(
                kt_num_subblocks >= 1 && kt_num_subblocks <= 4,
                "kt_num_subblocks must be 1-4 (sbh=1: Sk=8/16, sbh=2: Sk=4/8/16)");

            if constexpr (sbh == 1) {
                // sbh=1: split-matmul overlap (inner dim split across kt_num_subblocks)
                constexpr uint32_t matmul_inner = qktv_in0_block_w / kt_num_subblocks;
                for (uint32_t kt_sub = 0; kt_sub < kt_num_subblocks; ++kt_sub) {
                    // sub_exp drain in-place on cb_qkt_im
                    sub_exp_block_bcast_cols<PROFILING_ENABLED, scale_fp32, sbh, qkt_subblock_w, true>(
                        cb_qkt_im, cur_max, cur_sum, Sk_chunk_t, q_num_subblocks - 1, kt_sub);

                    // Before first matmul: wait for softmax'd QKT tiles and V tiles.
                    // V wait deferred from Phase 2 start — the drain loop above
                    // doesn't touch V, so the reader's V DMA overlaps with it.
                    if (kt_sub == 0) {
                        cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);
                        cb_wait_front(cb_v_in, Sk_chunk_t * vDHt);
                    }
                    if (kt_sub > 0) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }

                    // Matmul — FPU overlaps with SFPU EXP
                    {
                        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "QKT@V MM+Pack");
                        uint32_t v_index_offset = 0;
#ifdef ARCH_BLACKHOLE
                        mm_no_mop_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, matmul_inner);
#else
                        mm_block_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, matmul_inner);
#endif
                        for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                            blocked_matmul_and_pack<false, qktv_subblock_w, qktv_subblock_h, matmul_inner, vDHt, vDHt>(
                                cb_qkt_im,
                                cb_v_in,
                                cur_out,
                                qktv_in0_index_offset + kt_sub * matmul_inner,
                                kt_sub * matmul_inner * vDHt + v_index_offset,
                                0,
                                v_subblock * qktv_subblock_w);
                            v_index_offset += qktv_subblock_w;
                        }
                    }

                    if (kt_sub > 0) {
                        PACK((llk_pack_reconfig_l1_acc(0)));
                    }
                }
            } else {
                // sbh > 1: drain all sub_exp in-place, then full matmul (no split)
                // Can't split inner dim because matmul_block uses INNER_DIM as in0 row stride
                for (uint32_t kt_sub = 0; kt_sub < kt_num_subblocks; ++kt_sub) {
                    sub_exp_block_bcast_cols<PROFILING_ENABLED, scale_fp32, sbh, qkt_subblock_w, true>(
                        cb_qkt_im, cur_max, cur_sum, Sk_chunk_t, q_num_subblocks - 1, kt_sub);
                }

                cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);
                // V wait deferred from Phase 2 start — the drain loop above
                // doesn't touch V, so the reader's V DMA overlaps with it.
                cb_wait_front(cb_v_in, Sk_chunk_t * vDHt);
                {
                    MaybeDeviceZoneScopedN(PROFILING_ENABLED, "QKT@V MM+Pack");
                    uint32_t v_index_offset = 0;
#ifdef ARCH_BLACKHOLE
                    mm_no_mop_reinit_short(
                        cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, qktv_in0_block_w);
#else
                    mm_block_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, qktv_in0_block_w);
#endif
                    for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                        blocked_matmul_and_pack<false, qktv_subblock_w, qktv_subblock_h, qktv_in0_block_w, vDHt, vDHt>(
                            cb_qkt_im,
                            cb_v_in,
                            cur_out,
                            qktv_in0_index_offset,
                            v_index_offset,
                            0,
                            v_subblock * qktv_subblock_w);
                        v_index_offset += qktv_subblock_w;
                    }
                }
            }
            qktv_in0_index_offset += qktv_subblock_h * qktv_in0_block_w;
            qktv_in0_wait_tiles += qktv_in0_subblock_num_tiles;
        }

        // Per-row normalization lambda
        auto normalize_row = [&](uint32_t w_row, uint32_t& pushed) {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "ROW_NORM");
            cb_push_back(cur_sum, sbh);
            cb_push_back(cur_out, sbh * vDHt);
            normalize_row_streaming<PROFILING_ENABLED, sbh, vDHt>(
                cur_sum, cur_out, cb_col_identity, cb_recip_scratch, cb_normalized_out);
            pushed++;
        };

        // SALAD correction + optional normalization lambda
        // In ring_mode, skip per-row normalization on last iter — caller does bulk finalization.
        auto salad_correct_row = [&](uint32_t salad_row, uint32_t w_salad, bool last_iter, uint32_t& pushed) {
            {
                MaybeDeviceZoneScopedN(PROFILING_ENABLED, "S_SUM_CORR");
                mul_bcast_cols_l1_acc<sbh>(prev_sum, cb_exp_max_diff, cur_sum, salad_row, w_salad);
            }
            {
                MaybeDeviceZoneScopedN(PROFILING_ENABLED, "S_OUT_CORR");
                mul_block_bcast_cols_acc<sbh, vDHt>(prev_out, cb_exp_max_diff, cur_out, salad_row, w_salad);
            }
            if (last_iter && !ring_mode) {
                normalize_row(w_salad, pushed);
            }
        };

        // q_subblock 1..N-1: SALAD(prev) overlapped with matmul(cur)
        exp_packthread_tile_init<EXP_APPROX_MODE, false>();
        for (uint32_t q_subblock = 1; q_subblock < qktv_q_num_subblocks; ++q_subblock) {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Softmax(Q@KT)@V");
            uint32_t salad_row = q_subblock - 1;
            uint32_t w_salad = salad_row - pushed_rows;
            uint32_t w_q = q_subblock - pushed_rows;

            cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);

            if (!is_first_iter) {
                cb_reserve_back(cb_exp_max_diff, sbh);
                sub_exp_first_col_blocks<PROFILING_ENABLED, scale_fp32, sbh>(
                    prev_max, cur_max, cb_exp_max_diff, salad_row);
                cb_push_back(cb_exp_max_diff, sbh);
            }

            // Full matmul for current row
            {
                MaybeDeviceZoneScopedN(PROFILING_ENABLED, "QKT@V MM+Pack");
                uint32_t v_index_offset = 0;
                mm_block_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, qktv_in0_block_w);
                for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                    blocked_matmul_and_pack<false, qktv_subblock_w, qktv_subblock_h, qktv_in0_block_w, vDHt, vDHt>(
                        cb_qkt_im,
                        cb_v_in,
                        cur_out,
                        qktv_in0_index_offset,
                        v_index_offset,
                        w_q,
                        v_subblock * qktv_subblock_w);
                    v_index_offset += qktv_subblock_w;
                }
            }

            // SALAD corrections + optional per-row normalization
            if (!is_first_iter) {
                salad_correct_row(salad_row, w_salad, is_last_iter, pushed_rows);
            } else if (is_last_iter && !ring_mode) {
                normalize_row(w_salad, pushed_rows);
            }

            qktv_in0_index_offset += qktv_subblock_h * qktv_in0_block_w;
            qktv_in0_wait_tiles += qktv_in0_subblock_num_tiles;
        }

        // Pipeline drain: SALAD for last row
        {
            constexpr uint32_t salad_row = qktv_q_num_subblocks - 1;
            uint32_t w_salad = salad_row - pushed_rows;

            if (!is_first_iter) {
                cb_reserve_back(cb_exp_max_diff, sbh);
                sub_exp_first_col_blocks<PROFILING_ENABLED, scale_fp32, sbh>(
                    prev_max, cur_max, cb_exp_max_diff, salad_row);
                cb_push_back(cb_exp_max_diff, sbh);

                salad_correct_row(salad_row, w_salad, is_last_iter, pushed_rows);
            } else if (is_last_iter && !ring_mode) {
                normalize_row(w_salad, pushed_rows);
            }
        }

        // Bulk push — skip on last iteration when rows are consumed by normalization.
        // In ring_mode, always bulk push (even on last iter) — caller handles finalization.
        if (!is_last_iter || ring_mode) {
            cb_push_back(cur_sum, Sq_chunk_t);
            cb_push_back(cur_out, qktv_output_num_tiles);
        }

        cb_pop_front(cb_v_in, Sk_chunk_t * vDHt);
        cb_pop_front(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);
    }
}

/**
 * Streaming SDPA wrapper (v2): Q-chunk / K-chunk outer loop with ping-pong buffer management.
 * No row buffers — uses cb_push_back_hold_wr_ptr for direct cb_qkt_im writes.
 */
template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t Skt,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t subblock_h,
    bool use_padded_mask,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_normalized_out,
    uint32_t cb_mask_in>
void sdpa_standard_v2(
    const uint32_t q_chunks_per_core,
    const uint32_t k_num_chunks,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B) {
    for (uint32_t q = 0; q < q_chunks_per_core; q++) {
        // DeviceZoneScopedN("Q chunk");
        uint32_t alias_prev_sum = cb_sum_A, alias_cur_sum = cb_sum_B;
        uint32_t alias_prev_max = cb_max_A, alias_cur_max = cb_max_B;
        uint32_t alias_prev_out = cb_out_im_A, alias_cur_out = cb_out_im_B;

        // Per-iteration profiling via call_step: pass std::true_type{} to enable
        // MaybeDeviceZoneScopedN in sdpa_inner_loop_step, std::false_type{} to disable.
        // To profile specific iterations, change the gating condition below.
        auto call_step = [&](auto profiling_tag, bool is_last, bool is_first) {
            sdpa_inner_loop_step<
                decltype(profiling_tag)::value,
                Sq_chunk_t,
                Sk_chunk_t,
                Skt,
                DHt,
                vDHt,
                scale_fp32,
                subblock_h,
                use_padded_mask,
                false,  // ring_mode
                false,  // use_ring_mask
                cb_q_in,
                cb_kt_in,
                cb_v_in,
                cb_qkt_im,
                cb_identity_scale_in,
                cb_exp_max_diff,
                cb_col_identity,
                cb_recip_scratch,
                cb_normalized_out,
                cb_mask_in>(
                alias_prev_max,
                alias_cur_max,
                alias_prev_sum,
                alias_cur_sum,
                alias_prev_out,
                alias_cur_out,
                is_last,
                is_first);

            // Post-iteration cleanup
            if (!is_first) {
                cb_pop_front(cb_exp_max_diff, Sq_chunk_t);
                cb_pop_front(alias_prev_max, Sq_chunk_t);
                cb_pop_front(alias_prev_sum, Sq_chunk_t);
                cb_pop_front(alias_prev_out, Sq_chunk_t * vDHt);
            }

            if (is_last) {
                cb_pop_front(alias_cur_max, Sq_chunk_t);
            } else {
                std::swap(alias_prev_max, alias_cur_max);
                std::swap(alias_prev_sum, alias_cur_sum);
                std::swap(alias_prev_out, alias_cur_out);
            }
        };

        for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; k_chunk++) {
            // DeviceZoneScopedN("K chunk");
            bool is_first = (k_chunk == 0);
            bool is_last = (k_chunk == k_num_chunks - 1);

            // Default: profiling disabled. To profile specific iterations:
            // if (is_first) {
            //     call_step(std::true_type{}, is_last, is_first);
            // } else {
            //     call_step(std::false_type{}, is_last, is_first);
            // }
            call_step(std::false_type{}, is_last, is_first);
        }
        // Q already popped inside sdpa_inner_loop_step after Phase 1 of the last K chunk.
    }
}

/**
 * Streaming Ring SDPA (v2): Ring-aware variant of sdpa_standard_v2.
 * Calls sdpa_inner_loop_step with ring_mode=true and use_ring_mask=true.
 * After K-chunk loop: performs ring finalization (log, LSE, recip, sigmoid blend).
 *
 * Per-Q-chunk outer loop with ping-pong buffer management.
 * Per ring iteration, processes all Q chunks over [global_q_start, global_q_end).
 */
template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t Skt,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t subblock_h,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_mask_in,
    uint32_t cb_scale_in,
    uint32_t cb_lse_in,
    uint32_t cb_lse_out,
    uint32_t cb_prev_out,
    uint32_t cb_out>
void sdpa_ring_v2(
    const uint32_t global_q_start,
    const uint32_t global_q_end,
    const uint32_t num_kv_chunks,
    const uint32_t ring_iter,
    const uint32_t ring_id,
    const uint32_t num_local_k_chunks,
    const uint32_t local_padded_Nt,
    const uint32_t logical_nt,
    const bool ring_iter_needs_global_n_mask,
    const bool ring_iter_needs_joint_n_mask,
    const bool local_n_needs_masking,
    const uint32_t global_n_mask_chunk_id,
    const uint32_t local_n_mask_chunk_id,
    const uint32_t joint_n_mask_chunk_id,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B) {
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    uint32_t KV_chunks_processed_in_iter = 0;

    for (uint32_t q = global_q_start; q < global_q_end; q++) {
        uint32_t alias_prev_sum = cb_sum_A, alias_cur_sum = cb_sum_B;
        uint32_t alias_prev_max = cb_max_A, alias_cur_max = cb_max_B;
        uint32_t alias_prev_out = cb_out_im_A, alias_cur_out = cb_out_im_B;

        uint32_t KV_chunks_processed = 0;

        for (uint32_t k_chunk = 0; k_chunk < num_kv_chunks; ++k_chunk) {
            // Skip KV chunks beyond logical sequence length (non-joint only)
            const bool kv_chunk_is_joint = k_chunk >= num_local_k_chunks;
            const uint32_t kv_global_start_tile = local_padded_Nt * ring_id + k_chunk * Sk_chunk_t;
            if (!kv_chunk_is_joint && (kv_global_start_tile >= logical_nt)) {
                continue;
            }

            KV_chunks_processed++;
            KV_chunks_processed_in_iter++;

            bool is_first = (KV_chunks_processed == 1);

            // Determine if this K chunk needs masking
            bool apply_mask = (ring_iter_needs_global_n_mask && k_chunk == global_n_mask_chunk_id) ||
                              (local_n_needs_masking && k_chunk == local_n_mask_chunk_id) ||
                              (ring_iter_needs_joint_n_mask && (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id);

            // Call streaming step with ring_mode=true (skip normalization, always push)
            // is_last_iter=false: Q pop and finalization handled by this outer loop.
            sdpa_inner_loop_step<
                false,  // PROFILING_ENABLED
                Sq_chunk_t,
                Sk_chunk_t,
                Skt,
                DHt,
                vDHt,
                scale_fp32,
                subblock_h,
                false,  // use_padded_mask — ring uses ring mask instead
                true,   // ring_mode
                true,   // use_ring_mask
                cb_q_in,
                cb_kt_in,
                cb_v_in,
                cb_qkt_im,
                cb_identity_scale_in,
                cb_exp_max_diff,
                cb_col_identity,
                cb_recip_scratch,
                0,  // cb_normalized_out — unused in ring_mode (no per-row normalization)
                cb_mask_in>(
                alias_prev_max,
                alias_cur_max,
                alias_prev_sum,
                alias_cur_sum,
                alias_prev_out,
                alias_cur_out,
                false,  // is_last_iter
                is_first,
                apply_mask);

            // Post-iteration cleanup: pop previous values and swap aliases
            if (!is_first) {
                cb_pop_front(cb_exp_max_diff, Sq_chunk_t);
                cb_pop_front(alias_prev_max, Sq_chunk_t);
                cb_pop_front(alias_prev_sum, Sq_chunk_t);
                cb_pop_front(alias_prev_out, Sq_chunk_t * vDHt);
            }

            std::swap(alias_prev_max, alias_cur_max);
            std::swap(alias_prev_sum, alias_cur_sum);
            std::swap(alias_prev_out, alias_cur_out);
        }

        // Pop Q — not popped inside step since is_last_iter was always false
        cb_pop_front(cb_q_in, Sq_chunk_t * DHt);

        // ========== Ring Finalization ==========
        // After all K chunks: alias_prev_sum = final sum, alias_prev_max = final max,
        // alias_prev_out = final unnormalized output. alias_cur_* are empty.

        // 1. log(sum) → alias_cur_max (scratch)
        log_block(alias_prev_sum, alias_cur_max, Sq_chunk_t);

        // 2. LSE = scale * max + log(sum)
        mul_block_bcast_scalar_inplace<cb_scale_in, Sq_chunk_t>(alias_prev_max);
        add_block_inplace(alias_prev_max, alias_cur_max, Sq_chunk_t);

        // 3. Normalize output: out = out / sum
        recip_block_inplace(alias_prev_sum, Sq_chunk_t);
        mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(alias_prev_out, alias_prev_sum);

        if (ring_iter > 0) {
            // Sigmoid blend with previous ring iteration's output and LSE
            //   sig = sigmoid(cur_lse - prev_lse)
            //   out = prev_out - sig * (prev_out - cur_out)
            //   lse = prev_lse - logsigmoid(prev_lse - cur_lse)
            cb_wait_front(cb_lse_in, Sq_chunk_t);
            cb_wait_front(cb_prev_out, out_chunk_tiles);

            uint32_t alias_cur_lse = alias_prev_max;    // full (contains LSE)
            uint32_t alias_sig = alias_cur_max;         // empty
            uint32_t alias_cur_out_r = alias_prev_out;  // full (contains normalized output)
            uint32_t alias_sub = alias_cur_out;         // empty

            // sig = sigmoid(cur_lse - prev_lse)
            sigmoid_sub(alias_cur_lse, cb_lse_in, alias_sig, Sq_chunk_t);

            // sub = prev_out - cur_out
            reconfig_data_format(cb_prev_out, alias_cur_out_r);
            sub_block(cb_prev_out, alias_cur_out_r, alias_sub, out_chunk_tiles);
            // sub *= sig
            reconfig_data_format(alias_sub, alias_sig);
            mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_sub, alias_sig);
            // out = prev_out - sub
            reconfig_data_format(cb_prev_out, alias_sub);
            pack_reconfig_data_format(cb_out);
            sub_block(cb_prev_out, alias_sub, cb_out, out_chunk_tiles);
            cb_pop_front(cb_prev_out, out_chunk_tiles);
            cb_pop_front(alias_cur_out_r, out_chunk_tiles);
            cb_pop_front(alias_sub, out_chunk_tiles);

            // lse = prev_lse - logsigmoid(prev_lse - cur_lse)
            pack_reconfig_data_format(alias_sig);
            reconfig_data_format(cb_lse_in, alias_cur_lse);
            logsigmoid_sub(cb_lse_in, alias_cur_lse, alias_sig, Sq_chunk_t);
            sub_block(cb_lse_in, alias_sig, cb_lse_out, Sq_chunk_t);
            cb_pop_front(alias_sig, Sq_chunk_t);
            cb_pop_front(alias_cur_lse, Sq_chunk_t);
            cb_pop_front(cb_lse_in, Sq_chunk_t);
        } else {
            // First ring iteration: copy output and LSE directly
            pack_reconfig_data_format(cb_out);
            copy_block(alias_prev_out, cb_out, out_chunk_tiles);

            pack_reconfig_data_format(cb_lse_out);
            copy_block(alias_prev_max, cb_lse_out, Sq_chunk_t);
        }
    }

    // Dummy KV pop for double-buffer alignment (same as sdpa_inner_loop for RING)
    if (KV_chunks_processed_in_iter % 2 == 0) {
        cb_wait_front(cb_kt_in, DHt * Sk_chunk_t);
        cb_pop_front(cb_kt_in, DHt * Sk_chunk_t);
        cb_wait_front(cb_v_in, Sk_chunk_t * vDHt);
        cb_pop_front(cb_v_in, Sk_chunk_t * vDHt);
    }
}
